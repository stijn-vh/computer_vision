from calibration import Calibration
from background_substraction import BackgroundSubstraction
from voxel_reconstruction import VoxelReconstruction
import numpy as np

if __name__ == '__main__':   
    #cali = Calibration()
    #intrinsics = cali.obtain_intrinsics_from_cameras()
    #cali.obtain_extrinsics_from_cameras()

    S = BackgroundSubstraction()
    cam_means, cam_std_devs = S.create_background_model()
    thresholds = np.array([[10, 2, 18],
                           [10, 2, 14], #very short flicker
                           [10, 1, 10],
                           [10, 2, 22]])
    num_contours = [1, 2, 2, 1]
    show_video = False
    masks = S.background_subtraction(thresholds, num_contours, cam_means, cam_std_devs, show_video)

    VR = VoxelReconstruction('cameras.pickle')
    VR.run_voxel_reconstruction(masks)