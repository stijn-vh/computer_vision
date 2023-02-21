from calibration import Calibration
from voxel_reconstruction import VoxelReconstruction
from background_substraction import BackgroundSubstraction

if __name__ == '__main__':
    cali = Calibration()
    cali.load_frames_from_videos()