from background_substraction import BackgroundSubstraction
from voxel_reconstruction import VoxelReconstruction
from clustering import Clustering

import numpy as np
import pickle
import pandas as pd
from calibration import Calibration
import cv2 as cv
import json

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def save_to_json(name, object):
    with open(name + '.json', 'w') as handle:
        json.dump(object, handle, cls=NumpyEncoder)

def load_from_json(name):
    with open(name + '.json') as handle:
        data = json.load(handle)
    
    return data
    
def pickle_object(name, object):
    with open(name + '.pickle', 'wb') as handle:
        pd.to_pickle(name)

def load_pickle_object(name):
    with open(name + '.pickle', 'rb') as handle:
        return pickle.load(handle)
    
def determine_camera_params():
    cali = Calibration()
    cali.cameras = load_pickle_object('scaled_camera')
    intrinsics = cali.obtain_intrinsics_from_cameras()
    cali.obtain_extrinsics_from_cameras()
    pickle_object("scaled_camera", cali.cameras)

def determine_new_masks(show_video = True):
    S = BackgroundSubstraction()
    cam_means, cam_std_devs = S.create_background_model()
    # thresholds = np.array([[10, 2, 18],
    #                        [10, 2, 14],
    #                        [10, 1, 10], #glitchy for cam3
    #                        [10, 2, 22]]) #flickkery for cam 4 with body parts half missing
    # num_contours = [4, 5, 5, 4]
    thresholds = np.array([[10, 2, 18],
                           [10, 2, 14],
                           [7, 1, 14],
                           [7, 2, 20]])
    num_contours = [4, 5, 5, 6]
    # #Penalizing 2x more if pixel not in groundtruth but is in mask. fixed numcontours to equal 1
    # thresholds = np.array([[10, 1, 14],
    #                        [10, 2, 14],
    #                        [10, 1, 10],
    #                        [10, 2, 8]])
    # num_contours = [1, 2, 2, 1]
    #Penalizing 3x more if pixel not in groundtruth but is in mask. fixed numcontours to equal 1
    # thresholds = np.array([[2, 6, 12],
    #                        [2, 6, 14],
    #                        [4, 6, 20],
    #                        [1, 7, 14]])
    # num_contours = [1, 2, 1, 2]
    masks, frames = S.background_subtraction(thresholds, num_contours, cam_means, cam_std_devs, show_video)
    return masks, frames


def determine_new_thresholds():
    S = BackgroundSubstraction()
    cam_means, cam_std_devs = S.create_background_model()
    thresh, num_contours = S.gridsearch(cam_means, cam_std_devs)
    print("thresholds = ", thresh, "Num_contours =", num_contours)

def show_four_images(images):
    concat_row_1 = np.concatenate((images[0], images[1]), axis=0)
    concat_row_2 = np.concatenate((images[2], images[3]), axis=0)
    all_images = np.concatenate((concat_row_1, concat_row_2), axis=1)

    cv.imshow('all_images', all_images)

    cv.waitKey(0)

if __name__ == '__main__':
    #determine_camera_params()
    #determine_new_thresholds()
    print("creating masks")
    masks, frames = determine_new_masks(show_video=False)
    print('determined masks')
    save_to_json('masks', masks)
    save_to_json('frames', frames)
    print('saved masks')
    masks = np.array(load_from_json('masks'))
    print('loaded masks')
    VR = VoxelReconstruction('scaled_camera.pickle')
    #
    print('create lookup')
    lookup_table = VR.create_lookup_table()
    print('created')
    save_to_json('lookup_table', lookup_table)
    print('saved lookup')
    lookup_table = load_from_json('lookup_table')
    print('loaded lookup')
    # print('start reconstruction')
    VR.lookup_table = lookup_table
    print('start reconstruction')
    #VR.run_voxel_reconstruction(masks)
    print('done reconstruction')

    c = Clustering()
    c.cluster(VR.all_vis_voxels)

