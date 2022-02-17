import cv2
import numpy as np
import tkinter as tk
from tkinter.filedialog import askopenfilename
import matplotlib.pyplot as plt
from LeafArea import *

def click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pt = [x,y]
        refPt.append(pt)
    return refPt

# Importing image
root = tk.Tk()
root.update()
filename = askopenfilename(filetypes=[("images",["*.jpg", "*.jpeg"])])
object_img = cv2.imread(filename)
root.destroy()

scale_size = 20
refPt = list()

# Picking reference point for distortion correction
img_to_show = object_img.copy()
img2show = scale_img(img_to_show, scale_size)
cv2.imshow('Base Img', img2show)
cv2.setMouseCallback("Base Img", click)

while True:
    img_to_show = object_img.copy()
    img2show = scale_img(img_to_show, scale_size)

    if len(refPt) > 0:
        for pt in refPt:
            cv2.circle(img2show, tuple(pt), 3, (0, 0, 255), 3)
    cv2.imshow('Base Img', img2show)

    key = cv2.waitKey(25)
    if key == ord('q') or key == ord('Q'):
        break
    if key == ord('r') or key == ord('R'):
        refPt = list()
        img2show = scale_img(img_to_show, scale_size)
    if key == ord('c') or key == ord('R'):
        refPt.pop()
        img2show = scale_img(img_to_show, scale_size)

cv2.destroyAllWindows()

# Default paper size for measurement relative to paper size
wp = 210
hp = 297
scale_factor = 7

refPt = np.array(refPt) // (scale_size/100) # back to original size
reordered = reorder(refPt)
warp_img = warpImg(object_img, reordered, scale_factor*wp, scale_factor*hp)

segment_leaf = {
    'min_H':20, 'max_H':120,
    'min_S':30, 'max_S':255,
    'min_V':0, 'max_V':190}

segment_box = {
    'min_H':93, 'max_H':128,
    'min_S':15, 'max_S':255,
    'min_V':0, 'max_V':190}

# HSV segmentation
hsv_img = cv2.cvtColor(warp_img, cv2.COLOR_BGR2HSV)

box_mask = cv2.inRange(hsv_img, 
                   (segment_box['min_H'], segment_box['min_S'], segment_box['min_V']),
                   (segment_box['max_H'], segment_box['max_S'], segment_box['max_V']))

leaf_mask = cv2.inRange(hsv_img, 
                   (segment_leaf['min_H'], segment_leaf['min_S'], segment_leaf['min_V']),
                   (segment_leaf['max_H'], segment_leaf['max_S'], segment_leaf['max_V']))

box_mask = process_mask(box_mask)
leaf_mask = process_mask(leaf_mask)

img_to_show = warp_img.copy()
gamma = 10

# Calculate leaf area relative to segmented reference box
box_contour, _ = cv2.findContours(box_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
cBox = max(box_contour, key=cv2.contourArea)
box_area = cv2.contourArea(cBox)
(x, y, w, h) = cv2.boundingRect(cBox)
cv2.drawContours(img_to_show, cBox, -1, (255,0,0), 3)

leaf_contour, _ = cv2.findContours(leaf_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
cLeaf = max(leaf_contour, key=cv2.contourArea)
leaf_area = cv2.contourArea(cLeaf)
(x, y, w, h) = cv2.boundingRect(cLeaf)
cv2.drawContours(img_to_show, cLeaf, -1, (0,255,0), 3)

# Calculate leaf area relative to paper size
real_paper_area = wp*hp / 100
pixel_paper_area = warp_img.shape[0] * warp_img.shape[1]
real_area_paper = (real_paper_area/pixel_paper_area) * leaf_area
real_area_box = (1/box_area) * leaf_area

cv2.putText(img_to_show,
            f'leaf area paper: {round(real_area_paper,2)} cm2',
            (x-50,y+h+50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1, (0,0,255),2)
cv2.putText(img_to_show,
            f'leaf area box: {round(real_area_box,2)} cm2',
            (x-50,y+h+100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1, (255,0,0),2)

# Visualize segmented leaf
img2show = scale_img(img_to_show, 50)
mask2show = scale_img(leaf_mask, 50)

print(f'Filename\t:{filename[-16:-4]}')
print(f'leaf area paper\t: {round(real_area_paper,2)} cm2')
print(f'leaf area box\t: {round(real_area_box,2)} cm2')
plt.figure(figsize=(10,15))
plt.imshow(cv2.cvtColor(img2show, cv2.COLOR_BGR2RGB))
plt.show()