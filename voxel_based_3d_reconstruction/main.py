from calibration import Calibration
from background_substraction import VoxelReconstruction
from voxel_reconstruction import BackgroundSubstraction

if __name__ == '__main__':
    cali = Calibration()
    intrinsics = cali.obtain_intrinsics_from_cameras()
    cali.obtain_extrinsics_from_cameras()