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

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.


# Create window with provided image
def show_image(img, title='Current Image'):
    cv.namedWindow(image_name, cv.WINDOW_KEEPRATIO)
    cv.imshow(image_name, img)
    cv.setWindowTitle(image_name, title)
    cv.resizeWindow(image_name, 1900, 1080)

# Draw and show chessboardcorners based on found corners in image
def draw_chessboard_corners(corners, current_image, time):
    cv.drawChessboardCorners(current_image, (num_cols, num_rows), corners, True)
    show_image(current_image)
    cv.waitKey(time)


# Handle click event: if left mouse button is clicked on image, add coordinates of mouse to cornerspoint
def click_event(event, x, y, flags, params):
    current_image = params
    if event == cv.EVENT_LBUTTONDOWN and len(corner_points) < 4:
        print('new cornerpoint added: (' + str(x) + ', ' + str(y) + ')')
        corner_points.append([x, y])

        cv.circle(current_image, (x, y), radius=6, color=(0, 0, 255), thickness=1)
        show_image(current_image)



# Determine coordinates of next corner based on current coordinates and amount of columns/rows
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

# Show image, wait for 4 user-clicks which indicate the corners of the chessboard. 
# if 4 corners are specified, try to interpolate between the corner points
def determine_points_mannually(gray):
    show_image(gray, title="Choose points in Z pattern starting at the upper left")
    cv.setMouseCallback(image_name, click_event, gray)

    while 1:
        cv.waitKey(0)
        count_points = len(corner_points)

        if count_points == 4:
            return interpolate_four_corners(corner_points)
        else:
            print('Only ' + str(count_points) + ' added, please add ' + str(4 - count_points) + ' more')


# Improve image by making it black/white, adding blur and then smoothing it
def improve_image(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    smoothed = cv.GaussianBlur(gray, (0, 0), 3)
    return cv.addWeighted(gray, 2, smoothed, -1, 0)


# Try to find corners on image, 
#   if corners not found, try to do it manually
# then try to place corners more precise and show the result on the image
def handle_image(img, time = 50):
    global corner_points

    improved_gray = improve_image(img)
    ret, corners = cv.findChessboardCorners(improved_gray, (num_cols, num_rows), None)

    if ret == False:
        corners = determine_points_mannually(improved_gray)
        time = 2000

    improved_corners = cv.cornerSubPix(improved_gray, corners, (3, 3), (-1, -1), criteria)
    draw_chessboard_corners(improved_corners, img, time)

    objpoints.append(objp)
    imgpoints.append(improved_corners)
    corner_points = []

# Loop through all provided image paths and determine their cornerpoints, 
# then based on all the cornerpoints, calibrate the camera
def calibrate_on_images(images):
    for img_path in images:
        current_image = cv.imread(img_path)
        handle_image(current_image)

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, current_image.shape[0:2][::-1], None, None)

    cv.destroyAllWindows()

    return {'ret': ret, 'mtx': mtx, 'dist': dist}


# Run 1: use all training images (including the images with manually provided corner points)
def phase_1():
    auto_images = glob.glob('images/automatic/*.jpg')
    manual_images = glob.glob('images/manual/*.jpg')
    all_images = auto_images + manual_images

    return calibrate_on_images(all_images)


# Run 2:  use only ten images for which corner points were found automatically
def phase_2():
    auto_images = glob.glob('images/test/*.jpg')
    return calibrate_on_images(auto_images)


# Run 3: use only five out of the ten images in Run 2. In each run, you will calibrate the camera
def phase_3():
    auto_images = glob.glob('images/automatic/*.jpg')[:5]

    return calibrate_on_images(auto_images)


# Execute all runs in order and return list of params to main
def execute_offline_phase():
    results = phase_1()

    print(results['mtx'])
    print(results['ret'])

    return results


def set_config(c):
    global criteria, num_cols, num_rows, objp, image_name, corner_points
    criteria = c['criteria']
    num_cols = c['num_cols']
    num_rows = c['num_rows']
    image_name = c['image_name']
    objp = c['objp']
    corner_points = []
