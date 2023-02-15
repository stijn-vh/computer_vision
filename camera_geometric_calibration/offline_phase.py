import copy
import os
import numpy as np
import cv2 as cv
import glob

# Sources:
# https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html


# From slides:
# Workflow offline:
# • Print checkerboard on piece of paper (or take a real one)
# • Measure the stride length and fill it in
# • Take a number of pictures with the camera
# • Determine camera parameters using OpenCV functions (browse!)
# • Now your camera is calibrated



corner_points = []


def draw_chessboard_corners(corners, gray, ret, current_image):
    # Draw and display the corners
    # corners = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    cv.drawChessboardCorners(current_image, (num_cols, num_rows), corners, ret)
    cv.imshow('current_image', current_image)
    cv.waitKey(50)


def click_event(event, x, y, flags, params):
    current_image = params
    if event == cv.EVENT_LBUTTONDOWN and len(corner_points) < 4:
        print('new cornerpoint added: (' + str(x) + ', ' + str(y) + ')')
        corner_points.append((x, y))

        cv.circle(current_image, (x, y), radius=6, color=(0, 0, 255), thickness=1)
        cv.imshow('current_image', current_image)


def direction_step(p1, p2, num_steps):
    x1, y1 = p1
    x2, y2 = p2
    return (x2 - x1) / num_steps, (y2 - y1) / num_steps


def interpolate_four_corners(four_corners):
    upper_left_cor, upper_right_cor, lower_left_cor, lower_right_cor = four_corners
    ur_corx, ur_cory = upper_right_cor
    ul_corx, ul_cory = upper_left_cor

    # direction is also scaled to stepsize.
    rrow_dirx, rrow_diry = direction_step(upper_right_cor, lower_right_cor, (num_rows - 1))
    lrow_dirx, lrow_diry = direction_step(upper_left_cor, lower_left_cor, (num_rows - 1))

    corners = np.zeros((num_rows * num_cols, 1, 2))
    index = 0

    for i in range(num_rows):
        rrowptx = ur_corx + i * rrow_dirx
        rrowpty = ur_cory + i * rrow_diry
        lrowptx = ul_corx + i * lrow_dirx
        lrowpty = ul_cory + i * lrow_diry
        col_dirx, col_diry = direction_step((lrowptx, lrowpty), (rrowptx, rrowpty), (num_cols - 1))
        for j in range(num_cols):
            corners[index] = [lrowptx + j * col_dirx, lrowpty + j * col_diry]
            index += 1
    return np.float32(corners)


def determine_points_mannually(current_image):
    cv.imshow('current_image', current_image)
    cv.setMouseCallback('current_image', click_event, current_image)

    while 1:
        cv.waitKey(0)
        count_points = len(corner_points)
        if count_points == 4:
            return interpolate_four_corners(corner_points)
        else:
            print('Only ' + str(count_points) + ' added, please add ' + str(4 - count_points) + ' more')


def handle_image(current_image):
    global corner_points
    gray = cv.cvtColor(current_image, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (num_cols, num_rows), None)
    print(ret)
    # If found, add object points, image points (after refining them)
    if ret == False:
        corners = determine_points_mannually(current_image)
        ret = True
    draw_chessboard_corners(corners, gray, ret, current_image)
    corner_points = []
    return corners


def calibrate_on_images(images):
    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    for i,img_path in enumerate(images):
        image = cv.imread(img_path)
        corners = handle_image(image)
        if i == 0:
            rms, mtx, dist, rvecs, tvecs = cv.calibrateCamera([objp], [corners], image.shape[0:2][::-1], None, None)
            prevrms = rms
            prevmtx = mtx
            prevdist = dist
            # objpoints.append(objp)
            # imgpoints.append(corners)
        else:
            mtx = copy.deepcopy(prevmtx) #needed because these values themselves are changed
            dist = copy.deepcopy(prevdist)
            rms, mtx, dist, rvecs, tvecs = cv.calibrateCamera([objp], [corners], image.shape[0:2][::-1], mtx, dist)
            if rms < prevrms:
                prevrms = rms
                prevmtx = mtx
                prevdist = dist
                # objpoints.append(objp)
                # imgpoints.append(corners)
    cv.destroyAllWindows()

    return {'mtx': prevmtx, 'dist': prevdist}


# Run 1: use all training images (including the images with manually provided corner points)
def phase_1():
    auto_images = glob.glob('images/automatic/*.jpg')
    manual_images = glob.glob('images/manual/*.jpg')
    all_images = auto_images + manual_images

    return calibrate_on_images(all_images)


# Run 2:  use only ten images for which corner points were found automatically
def phase_2():
    auto_images = glob.glob('images/automatic/*.jpg')[:10]

    return calibrate_on_images(auto_images)


# Run 3: use only five out of the ten images in Run 2. In each run, you will calibrate the camera
def phase_3():
    auto_images = glob.glob('images/automatic/*.jpg')[:5]

    return calibrate_on_images(auto_images)


# Execute all runs in order and return list of params to main
def execute_offline_phase():
    phase_1_results = phase_2()
    return phase_1_results


def set_config(c):
    global criteria, num_cols, num_rows, objp
    criteria = c['criteria']
    num_cols = c['num_cols']
    num_rows = c['num_rows']
    objp = np.zeros((num_cols * num_rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:num_cols, 0:num_rows].T.reshape(-1, 2)
