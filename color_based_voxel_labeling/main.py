from background_substraction import BackgroundSubstraction
from voxel_reconstruction import VoxelReconstruction
from color_models import ColourModels
from clustering import Clustering

import numpy as np
import pickle
import pandas as pd
from calibration import Calibration
import cv2 as cv
import json

import assignment as Assignment
import executable as Executable


import os


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

BS = None
VR = None
C = None
CM = None

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


def determine_new_masks(show_video=True):
    S = BackgroundSubstraction()
    cam_means, cam_std_devs = S.create_background_model()
    # thresholds = np.array([[10, 2, 18],
    #                        [10, 2, 14],
    #                        [10, 1, 10], #glitchy for cam3
    #                        [10, 2, 22]]) #flickkery for cam 4 with body parts half missing
    # num_contours = [4, 5, 5, 4]
    thresholds = np.array([[10, 2, 18],
                           [10, 2, 14],
                           [10, 1, 14],
                           [10, 2, 20]])
    num_contours = [4, 5, 5, 6]
    # #Penalizing 2x more if pixel not in groundtruth but is in mask. fixed numcontours to equal 1
    # thresholds = np.array([[10, 1, 14],
    #                        [10, 2, 14],
    #                        [10, 1, 10],
    #                        [10, 2, 8]])
    # num_contours = [1, 2, 2, 1]
    # Penalizing 3x more if pixel not in groundtruth but is in mask. fixed numcontours to equal 1
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


# def handle_frame(frame, cam):


#     # Loop door alle frames
#         # Per camera:
#             # 1. Determine mask
#             # 2. Determine Hue, Saturation, Value
#         # 3. Voxel Reconstruct Frame
#         # 4. Offline cluster: find 4 cluster in voxels (in frame where vo) -> sla op
#         # 5. Clustering
#         # 6. Colour models

def load_parameters():
    parameters = {
        'rotation_vectors': [], 'translation_vectors': [], 'intrinsics': [], 'dist_mtx': [],
        'stepsize': 4,
        'amount_of_frames': 400,
        'cam_numbers': 4,
        'path': 'scaled_camera.pickle'
    }

    with open(parameters['path'], 'rb') as f:
        camera_params = pickle.load(f)

        for camera in camera_params:
            parameters['rotation_vectors'].append(np.array(camera_params[camera]['extrinsic_rvec']))
            parameters['translation_vectors'].append(np.array(camera_params[camera]['extrinsic_tvec']))
            parameters['intrinsics'].append(camera_params[camera]['intrinsic_mtx'])
            parameters['dist_mtx'].append(camera_params[camera]['intrinsic_dist'])

    return parameters


def init_models(params):
    global C, CM, VR, BS

    C = Clustering()
    BS = BackgroundSubstraction()
    BS.create_background_model()

    CM = ColourModels(params)

    # four_good_offline_voxel_clusters_per_camera = load_from_json()
    # corresponding_frame_per_camera = load_from_json()
    CM = ColourModels(params)
    # CM.create_offline_model(four_good_offline_voxel_clusters_per_camera, corresponding_frame_per_camera)

    VR = VoxelReconstruction(params)

def handle_frame(videos, cam_numbers, frame_number, prev):
    global C, CM, VR, BS

    cameras_masks = []
    cameras_frames = []

    for i in range(cam_numbers):
        ret, frame = BS.read_video(videos[i])

        cameras_frames.append(frame)
        cameras_masks.append(BS.compute_mask_in_frame(frame, i))

    if frame_number == 0:
        voxels = VR.reconstruct_voxels(cameras_masks, None, frame_number)
    else:
        voxels = VR.reconstruct_voxels(cameras_masks, prev, frame_number)

    Assignment.voxels_per_frame.append(voxels)

    voxel_clusters, cluster_centres, compactness  = C.cluster(voxels)
    matching = CM.matching_for_frame(voxel_clusters, cameras_frames)  # matching[i][j] = 1 if cluster j belongs to model i

    return cameras_masks

def handle_videos(params):
    global C, CM, VR, BS

    # print('start creation')
    # lookup_table = VR.create_lookup_table()
    # print('start saving to json')
    # save_to_json("lookup_table_"+ str(params['stepsize']), lookup_table)
    # print('end')
    print('start loading from json')
    VR.lookup_table = load_from_json('lookup_table_' + str(params['stepsize']))
    print('done loading json')

    videos = []

    for i in range(params['cam_numbers']):
        videos.append(cv.VideoCapture(os.path.dirname(__file__) + "\\data\\cam" + str(i + 1) + "\\video.avi"))

    prev_cameras_masks = []

    for frame_number in range(params['amount_of_frames']):
        prev_cameras_masks = handle_frame(videos, params['cam_numbers'], frame_number, prev_cameras_masks)

        #add cluster centres with their matching to a list
    # call a plot function which plots the different cluster centres and colours them according to their matching
    Executable.main()


if __name__ == '__main__':
    params = load_parameters()

    init_models(params)

    handle_videos(params)
