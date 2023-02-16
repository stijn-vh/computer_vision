import cv2 as cv
import numpy as np
import glob


# From slides:
# Workflow online:
# • Read an image/camera frame
# • Draw a box on a detected chessboard in the right perspective

# Create window with provided image
def show_image(img):
    cv.namedWindow(image_name, cv.WINDOW_KEEPRATIO)
    cv.imshow(image_name, img)
    cv.resizeWindow(image_name, 1900, 1080)

# Cast provided points to int's and return in typle-form
def get_point_tuple(pts):
    return tuple(map(int, pts.ravel()))


# Get coordinates for direction of the world 3D axes and draw a line between the origin and these coordinates
def draw_axes_on_image(img, imgpts):
    origin = get_point_tuple(imgpts[0])

    imgpx = get_point_tuple(imgpts[1])
    imgpy = get_point_tuple(imgpts[2])
    imgpz = get_point_tuple(imgpts[3])

    img = cv.line(img, origin, imgpx, (255, 0, 0), 5)
    img = cv.line(img, origin, imgpy, (0, 255, 0), 5)
    img = cv.line(img, origin, imgpz, (0, 0, 255), 5)

    return img


# Draw contours and lines of a cube on image based on provided image points
def draw_cube_on_image(img, imgpts):
    imgpts = np.array(list(map(get_point_tuple, imgpts)))
    img = cv.drawContours(img, [imgpts[:4]], -1, (120, 120, 0), 3)

    for i, j in zip(range(4), range(4, 8)):
        img = cv.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (120, 120, 0), 3)

    img = cv.drawContours(img, [imgpts[4:]], -1, (120, 120, 0), 3)
    return img

# Improve found corners, find correct rotation and translation vectors,
# project points from 3D to image based on found matrices and draw axes/cube on image
def project_points(gray, corners, axis, cube, estimated_camera_params):
    corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    ret, rvec, tvec = cv.solvePnP(objp, corners2, estimated_camera_params['mtx'], estimated_camera_params['dist'])

    axpts, jac = cv.projectPoints(axis, rvec, tvec, estimated_camera_params['mtx'], estimated_camera_params['dist'])
    cubepts, _ = cv.projectPoints(cube, rvec, tvec, estimated_camera_params['mtx'], estimated_camera_params['dist'])

    img = draw_axes_on_image(img, axpts, get_point_tuple(corners2[0]))
    img = draw_cube_on_image(img, cubepts, get_point_tuple(corners2[0]))

    return img

# Find corners of image, project points from 3D to image plane and draw axes/cube
def handle_image(img, estimated_camera_params):
    axsize = 6
    cubesize = 4

    axis = np.float32([[0, 0, 0], [axsize, 0, 0], [0, axsize, 0], [0, 0, -axsize]])
    cube = np.float32([[0, 0, 0], [cubesize, 0, 0], [cubesize, cubesize, 0], [0, cubesize, 0], [0, 0, -cubesize],
                       [cubesize, 0, -cubesize], [cubesize, cubesize, -cubesize], [0, cubesize, -cubesize]])

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, (num_cols, num_rows), None)

    if ret == True:
        img = project_points(gray, corners, axis, cube, estimated_camera_params)
        show_image(img)
        cv.waitKey(1)

# Start webcam, try to draw cube and axes on the 3D world origin on each frame
def draw_on_webcam(estimated_camera_params):
    cam = cv.VideoCapture(0)

    while (True):
        ret, frame = cam.read()
        show_image(frame)
        handle_image(frame, estimated_camera_params)

        key = cv.waitKey(1)

        if (key != -1):
            break

    return

# Try to draw cube and axes on test image
def draw_on_image(estimated_camera_params):
    test_image = cv.imread(glob.glob('images/test_image.jpg')[0])
    handle_image(test_image, estimated_camera_params)

# Perform online phase (draw cube and axes on image/webcam)
def execute_online_phase(estimated_camera_params):
    # draw_on_image(estimated_camera_params)
    draw_on_webcam(estimated_camera_params)

    cv.waitKey(0)

# Set configurations for online phase
def set_config(c):
    global criteria, num_cols, num_rows, objp, image_name
    criteria = c['criteria']
    num_cols = c['num_cols']
    num_rows = c['num_rows']
    image_name = c['image_name']
    objp = c['objp']
