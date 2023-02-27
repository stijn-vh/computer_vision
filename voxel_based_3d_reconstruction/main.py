from background_substraction import BackgroundSubstraction
from voxel_reconstruction import VoxelReconstruction
import numpy as np
import pickle
from calibration import Calibration

def pickle_object(name, object):
    with open(name + '.pickle', 'wb') as handle:
        pickle.dump(object, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_pickle_object(name):
    with open(name + '.pickle', 'rb') as handle:
        return pickle.load(handle)
    
def determine_camera_params():
    cali = Calibration()
    intrinsics = cali.obtain_intrinsics_from_cameras()
    cali.obtain_extrinsics_from_cameras()

def determine_new_masks():
    S = BackgroundSubstraction()
    cam_means, cam_std_devs = S.create_background_model()
    thresholds = np.array([[10, 2, 18],
                           [10, 2, 14], #very short flicker
                           [10, 1, 10],
                           [10, 2, 22]])
    num_contours = [1, 2, 2, 1]
    show_video = False

    return S.background_subtraction(thresholds, num_contours, cam_means, cam_std_devs, show_video)

if __name__ == '__main__':
    VR = VoxelReconstruction('cameras.pickle')
    #lookup_table = VR.create_lookup_table()
    #pickle_object('lookup_table', lookup_table)
    masks = load_pickle_object('masks')
    lookup_table = load_pickle_object('lookup_table')

    VR.lookup_table = lookup_table
    VR.run_voxel_reconstruction(masks)