import os
import numpy as np
import cv2 as cv
import glob

# From slides:
# Workflow offline: 
# • Print checkerboard on piece of paper (or take a real one)
# • Measure the stride length and fill it in
# • Take a number of pictures with the camera
# • Determine camera parameters using OpenCV functions (browse!)
# • Now your camera is calibrated

detectable_training_images = []
undetectable_training_images = []
all_training_images = np.concatenate((detectable_training_images, undetectable_training_images))

corner_points = []

def click_event(event, x, y, flags, params):
    current_image = params
    if event == cv.EVENT_LBUTTONDOWN:
        print('new cornerpoint added: (' + str(x) + ', ' + str(y) + ')')
        corner_points.append((x, y))

        cv.circle(current_image, (x,y), radius=6, color=(0, 0, 255), thickness=1)
        cv.imshow('current_image', current_image)
        
    if len(corner_points) == 4:
        return corner_points

def determine_points_mannually(current_image):
    cv.imshow('current_image', current_image)
    cv.setMouseCallback('current_image', click_event, current_image)

    cv.waitKey(0)
    cv.destroyAllWindows()

def handle_image(img_path, objp, criteria):
    current_image = cv.imread(img_path)
    gray = cv.cvtColor(current_image, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (9,6), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        # Draw and display the corners
        cv.drawChessboardCorners(current_image, (9,6), corners2, ret)
        cv.imshow('current_image', current_image)
        cv.waitKey(500)
    else:
        determine_points_mannually(current_image)
    
    return objp, corners2, gray

def geometric_camera_calibration():
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros((9*6,3), np.float32)
    objp[:,:2] = 24*np.mgrid[0:9,0:6].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    images = glob.glob('images/*.jpg')

    for fname in images:
        objp, imgp, gray = handle_image(fname, objp, criteria)
        objpoints.append(objp)
        imgpoints.append(imgp)

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    cv.destroyAllWindows()
    
# Run 1: use all training images (including the images with manually provided corner points)
def calibrate_on_all_images(self, images):
    return

# Run 2:  use only ten images for which corner points were found automatically
def calibrate_on_automatic_images(self, images, amount = 10):
    return

# Run 3: use only five out of the ten images in Run 2. In each run, you will calibrate the camera
def run_3(self):
    # Could maybe be done in function of Run 2?
    return

# Execute all runs in order and return list of params to main
def execute_offline_phase():
    geometric_camera_calibration()

execute_offline_phase()
