import cv2 as cv
import numpy as np
from background_substraction import BackgroundSubstraction
from sklearn.mixture import GaussianMixture

if __name__ == '__main__':
    S = BackgroundSubstraction()
    cam_means, cam_std_devs = S.create_background_model()
    # thresholds, num_contours = S.gridsearch(cam_means, cam_std_devs)
    thresholds = np.array([[10, 12, 12],
                           [2, 12, 16],
                           [4, 16, 24],
                           [2, 12, 24]])
    num_contours = [1, 2, 2, 1]
    S.background_subtraction(thresholds, num_contours, cam_means, cam_std_devs)


