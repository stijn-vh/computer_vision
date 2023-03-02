from background_substraction import BackgroundSubstraction
from voxel_reconstruction import VoxelReconstruction
import numpy as np
import pickle
from calibration import Calibration
import cv2 as cv

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
    pickle_object("scaled_camera", cali.cameras)

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


def show_four_images(images):
    concat_row_1 = np.concatenate((images[0], images[1]), axis=0)
    concat_row_2 = np.concatenate((images[2], images[3]), axis=0)
    all_images = np.concatenate((concat_row_1, concat_row_2), axis=1)

    cv.imshow('all_images', all_images)

    cv.waitKey(0)

if __name__ == '__main__':  
    determine_camera_params()
    VR = VoxelReconstruction('scaled_camera.pickle')

    print('create lookup')
    lookup_table = VR.create_lookup_table()
    pickle_object('lookup_table', lookup_table)
    print('done lookup')

    masks = load_pickle_object('masks')
    lookup_table = load_pickle_object('lookup_table_quick')

    #VR.lookup_table = lookup_table
    print('start reconstruction')
    #VR.test_voxel_reconstruction(masks)
    #VR.run_voxel_reconstruction(masks)