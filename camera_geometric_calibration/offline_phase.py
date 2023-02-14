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
num_cols = 9
num_rows = 6

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.
images = glob.glob('images/*.jpg')

corner_points = []

objp = np.zeros((num_cols*num_rows,3), np.float32)
objp[:,:2] = 24*np.mgrid[0:num_cols,0:num_rows].T.reshape(-1,2)

def draw_chessboard_corners(corners, gray, criteria, ret, current_image):
    # Draw and display the corners
    cv.drawChessboardCorners(current_image, (num_cols, num_rows), corners, ret)
    cv.imshow('current_image', current_image)
    cv.waitKey(50)

    objpoints.append(objp)
    imgpoints.append(corners)

def click_event(event, x, y, flags, params):
    current_image = params
    if event == cv.EVENT_LBUTTONDOWN and len(corner_points) < 3:
        print('new cornerpoint added: (' + str(x) + ', ' + str(y) + ')')
        corner_points.append((x, y))

        cv.circle(current_image, (x, y), radius=6, color=(0, 0, 255), thickness=1)
        cv.imshow('current_image', current_image)

def direction_step(p1, p2, num_steps):
    x1, y1 = p1
    x2, y2 = p2
    return (x2 - x1)/num_steps, (y2 - y1)/num_steps

def interpolate_three_corners(three_corners):
    # traverse rows towards the middle point, traverse cols away from middle point
    upper_right_cor, lower_right_cor, lower_left_cor = three_corners
    ur_corx, ur_cory = upper_right_cor

    # direction is also scaled to stepsize.
    row_dirx, row_diry = direction_step(upper_right_cor, lower_right_cor, (num_rows - 1))
    col_dirx, col_diry = direction_step(lower_right_cor, lower_left_cor, (num_cols - 1))

    index = 0
    corners = np.zeros((num_rows * num_cols, 1, 2))  # same shape as the findChessboardCorners output
    for j in range(num_rows):
        currentx = ur_corx + j * row_dirx
        currenty = ur_cory + j * row_diry
        for i in range(num_cols):
            corners[index] = [currentx, currenty]
            index += 1
            currentx += col_dirx
            currenty += col_diry

    return np.float32(corners)

def determine_points_mannually(current_image):
    cv.imshow('current_image', current_image)
    cv.setMouseCallback('current_image', click_event, current_image)

    while 1:
        cv.waitKey(0)
        count_points = len(corner_points)
        if count_points == 3:
            print("corner points = ", corner_points)
            print("interpolation = ", interpolate_three_corners(corner_points))
            return interpolate_three_corners(corner_points)
        else:
            print('Only ' + str(count_points) + ' added, please add ' + str(3 - count_points) + ' more')


def handle_image(img_path, criteria):
    current_image = cv.imread(img_path)
    gray = cv.cvtColor(current_image, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (num_cols, num_rows), None)

    # If found, add object points, image points (after refining them)
    if ret == False:
        corners = determine_points_mannually(current_image)
        ret = True

    draw_chessboard_corners(corners, gray, criteria, ret, current_image)

    return gray

def geometric_camera_calibration():
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros((num_cols * num_rows, 3), np.float32)
    objp[:, :2] = 24 * np.mgrid[0:num_cols, 0:num_rows].T.reshape(-1, 2)

    for fname in images:
        gray = handle_image(fname, criteria)

    # TODO takes gray of last image?
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    cv.destroyAllWindows()

    return {'mtx': mtx, 'dist': dist}

# Run 1: use all training images (including the images with manually provided corner points)
def calibrate_on_all_images(self, images):
    return

# Run 2:  use only ten images for which corner points were found automatically
def calibrate_on_automatic_images(self, images, amount=10):
    return

# Run 3: use only five out of the ten images in Run 2. In each run, you will calibrate the camera
def run_3(self):
    # Could maybe be done in function of Run 2?
    return

# Execute all runs in order and return list of params to main
def execute_offline_phase():
    return geometric_camera_calibration()
