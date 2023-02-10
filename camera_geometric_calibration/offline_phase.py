import numpy as np

class offline_phase:
    detectable_training_images = []
    undetectable_training_images = []
    all_training_images = np.concatenate((detectable_training_images, undetectable_training_images))
    test_image = None

    def geometric_camera_calibration():
        return
    
    # Run 1: use all training images (including the images with manually provided corner points)
    def calibrate_on_all_images(images):
        return

    # Run 2:  use only ten images for which corner points were found automatically
    def calibrate_on_automatic_images(images, amount = 10):
        return

    # Run 3: use only five out of the ten images in Run 2. In each run, you will calibrate the camera
    def run_3():
        # Could maybe be done in function of Run 2?
        return

    # Execute all runs in order and return list of params to main
    def execute_offline_phase():
        return
