import numpy as np
import cv2
import os

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
current_image = None

def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print('new cornerpoint added: (' + str(x) + ', ' + str(y) + ')')
        corner_points.append((x, y))

    if len(corner_points) == 4:
        return corner_points


def geometric_camera_calibration(self, click_event):
    return
    
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
    abs_path = r'C:\Users\fedor\Desktop\Computer Vision\computer_vision\camera_geometric_calibration\images\WIN_20230212_15_51_43_Pro.jpg'
    print(abs_path)
    current_image = cv2.imread(abs_path, 1)
    cv2.imshow('current_image', current_image)
    cv2.setMouseCallback('current_image', click_event)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

execute_offline_phase()
