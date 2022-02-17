import cv2
import numpy as np

# image processing methods
def scale_img(img, scale):
    scale_percent = scale # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return resized

def process_mask(mask):
    kernel = np.ones((11,11), np.uint8) 
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask

# warping image methods
def reorder(points):
    newPts = np.zeros_like(points)
    points = points.reshape((4,2))
    add = points.sum(1)
    newPts[0] = points[np.argmin(add)]
    newPts[3] = points[np.argmax(add)]
    diff = np.diff(points, axis=1)
    newPts[1] = points[np.argmin(diff)]
    newPts[2] = points[np.argmax(diff)]
    return newPts
    
def warpImg(img, points, w, h):
    pts1 = np.float32(points)
    pts2 = np.float32([[0,0], [w,0], [0,h], [w,h]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    img_warp = cv2.warpPerspective(img, matrix, (w,h))
    return img_warp
