import cv2 as cv
import numpy as np
import glob
# From slides: 
# Workflow online:
# • Read an image/camera frame
# • Draw a box on a detected chessboard in the right perspective

test_image = None

# take the test image and draw the world 3D axes (XYZ) with the origin at the center 
# of the world coordinates, using the estimated camera parameters
def draw_axes_on_image(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)

    return img

# draw a cube which is located at the origin of the world coordinates. 
# You can get bonus points for doing this in real time using your webcam
def draw_cube_on_image(image):
    return

def handle_image(img, estimated_camera_params):
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((9*6,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

    axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)

    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, (9,6),None)

    if ret == True:
        corners2 = cv.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        # Find the rotation and translation vectors.
        ret,rvecs, tvecs = cv.solvePnP(objp, corners2, estimated_camera_params['mtx'], estimated_camera_params['dist'])
        # project 3D points to image plane
        imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, estimated_camera_params['mtx'], estimated_camera_params['dist'])
        img = draw_axes_on_image(img, corners2, imgpts)
        cv.imshow('img',img)

def draw_cube_on_webcam(estimated_camera_params):
    images = glob.glob('images/*.jpg')

    for img in images:
        current_image = cv.imread(img)

        handle_image(current_image, estimated_camera_params)
    cam = cv.VideoCapture(0)

    while(True):
        ret, frame = cam.read()
        cv.imshow('webcam', frame)
        handle_image(frame)

        key = cv.waitKey(1)

        if (key != -1):
            break


def execute_online_phase(estimated_camera_params):
    draw_cube_on_webcam(estimated_camera_params)