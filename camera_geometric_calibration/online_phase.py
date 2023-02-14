import cv2 as cv
import numpy as np
import glob
# From slides: 
# Workflow online:
# • Read an image/camera frame
# • Draw a box on a detected chessboard in the right perspective

test_image = None

def get_point_tuple(pts):
    return tuple(map(int, pts.ravel()))

# take the test image and draw the world 3D axes (XYZ) with the origin at the center 
# of the world coordinates, using the estimated camera parameters
def draw_axes_on_image(img, corners, imgpts):
    corner = get_point_tuple(corners[0])

    imgp0 = get_point_tuple(imgpts[0])
    imgp1 = get_point_tuple(imgpts[1])
    imgp2 = get_point_tuple(imgpts[2])

    img = cv.line(img, corner, imgp0, (255,0,0), 5)
    img = cv.line(img, corner, imgp1, (0,255,0), 5)
    img = cv.line(img, corner, imgp2, (0,0,255), 5)

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
        cv.imshow('webcam',img)
        cv.waitKey(50)

def draw_cube_on_webcam(estimated_camera_params):
    cam = cv.VideoCapture(0)

    while(True):
        ret, frame = cam.read()
        cv.imshow('webcam', frame)
        handle_image(frame, estimated_camera_params)

        key = cv.waitKey(1)

        if (key != -1):
            break
    
    return

def execute_online_phase(estimated_camera_params):
    draw_cube_on_webcam(estimated_camera_params)