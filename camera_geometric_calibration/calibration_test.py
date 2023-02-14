# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 15:38:53 2023

@author: fedor
"""

import numpy as np
import cv2 as cv
import glob

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((9 * 6, 3), np.float32)

# Multiplied by 24 so that the object point coordinates are in mm
objp[:, :2] = 24 * np.mgrid[0:6, 0:9].T.reshape(-1, 2)
# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.
images = glob.glob('images/*.jpg')
for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners. shape is cols,rows
    ret, corners = cv.findChessboardCorners(gray, (6, 9), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        # Draw and display the corners
        cv.drawChessboardCorners(img, (6, 9), corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(1000)
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
cv.destroyAllWindows()

"""
img = cv.imread('images/WIN_20230212_15_51_43_Pro.jpg')
h,  w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
# undistort
dst = cv.undistort(img, mtx, dist, None, newcameramtx)
# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('calibresult.png', dst)
"""
import math

mean_error = 0
points = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)
    points += len(imgpoints2)
    mean_error += error

print("ret:", str(ret))
print("total error: {}".format(math.sqrt(mean_error / points)))


"""
    dist12 = euclid_distance(cor1, cor2)
    dist32 = euclid_distance(cor1, cor3)

    # Assumes that the corner points are further apart in the direction with more chessboard squares,
    # which will be true for normal inputs.
    if num_rows <= num_cols:
        if dist12 <= dist32:
            row_cor = cor1
            col_cor = cor3
        else:
            row_cor = cor3
            col_cor = cor1
    else:
        if dist12 <= dist32:
            col_cor = cor1
            row_cor = cor3
        else:
            col_cor = cor3
            row_cor = cor1
"""