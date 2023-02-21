import cv2 as cv
import numpy as np
from voxel_reconstruction import VoxelReconstruction
from sklearn.mixture import GaussianMixture

if __name__ == '__main__':
    V = VoxelReconstruction()
    cam_means, cam_std_devs = V.create_background_model()
    V.background_subtraction(cam_means, cam_std_devs)

