import cv2 as cv
import numpy as np
from background_substraction import BackgroundSubstraction
from sklearn.mixture import GaussianMixture

if __name__ == '__main__':
    S = BackgroundSubstraction()
    cam_means, cam_std_devs = S.create_background_model()
    #S.gridsearch(cam_means, cam_std_devs)
    # thresholds = np.array([[10, 12, 12],
    #                        [2, 12, 16], #no flickering
    #                        [4, 16, 24],
    #                        [2, 12, 24]])
    # num_contours = [1, 2, 2, 1]
    # thresholds = np.array([[10, 6, 10],
    #                        [10, 6, 12], #flickering cam 2
    #                        [10, 6, 8],
    #                        [10, 6, 22]])
    # num_contours = [1, 2, 2, 1]
    thresholds = np.array([[10, 2, 18],
                           [10, 2, 14], #very short flicker
                           [10, 1, 10],
                           [10, 2, 22]])
    num_contours = [1, 2, 2, 1]
    show_video = False
    S.background_subtraction(thresholds, num_contours, cam_means, cam_std_devs, show_video)


