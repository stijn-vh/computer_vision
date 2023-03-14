import copy

from background_substraction import BackgroundSubstraction
from voxel_reconstruction import VoxelReconstruction
from color_models import ColourModels
from clustering import Clustering
from trajectory_plotter import TrajectoryPlotter

import numpy as np
import pickle
import pandas as pd
from calibration import Calibration
import cv2 as cv
import json
from json_helper import JsonHelper

import assignment as Assignment
import executable as Executable

import os

BS = None
VR = None
C = None
CM = None
JH = None
TP = None


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
    thresholds = np.array([[10, 2, 18],
                           [10, 2, 14],
                           [10, 1, 14],
                           [10, 2, 20]])
    num_contours = [4, 5, 5, 6]
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


def save_offline_model_information(voxel_clusters, cameras_frames, cameras_framesBGR, frame_number):
    if frame_number == 0:
        JH.save_to_json("clusters_cam_1_2_4", voxel_clusters)
        JH.save_to_json("cameras_frames_1_2_4", cameras_frames)
        CM.plot_projected_voxels(voxel_clusters, cameras_framesBGR[0], 0)
        CM.plot_projected_voxels(voxel_clusters, cameras_framesBGR[1], 1)
        CM.plot_projected_voxels(voxel_clusters, cameras_framesBGR[3], 3)
        print("done frame 0")
    if frame_number == 180:
        JH.save_to_json("clusters_cam_3", voxel_clusters)
        JH.save_to_json("cameras_frames_3", cameras_frames)
        CM.plot_projected_voxels(voxel_clusters, cameras_framesBGR[2], 2)
        print("done frame 180")


def load_parameters():
    parameters = {
        'rotation_vectors': [], 'translation_vectors': [], 'intrinsics': [], 'dist_mtx': [],
        'stepsize': 4,
        'amount_of_frames': 200,
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
    global C, CM, VR, BS, JH, TP

    JH = JsonHelper()
    C = Clustering()
    BS = BackgroundSubstraction()
    BS.create_background_model()

    CM = ColourModels(params)
    CM.load_create_offline_model()

    VR = VoxelReconstruction(params)
    TP = TrajectoryPlotter((VR.xb, VR.yb))
    # print('start creation')
    # lookup_table = VR.create_lookup_table()
    # print('start saving to json')
    # save_to_json("lookup_table_"+ str(params['stepsize']), lookup_table)
    # print('end')
    print('start lookup table loading from json')
    VR.lookup_table = JH.load_from_json('lookup_table_' + str(params['stepsize']))
    print('done loading json')


def determine_cameras_masks_frames(cam_numbers, videos):
    cameras_masks = []
    cameras_frames = []
    cameras_framesBGR = []

    for cam in range(cam_numbers):
        ret, frameBGR = videos[cam].read()
        frame = np.float32(cv.cvtColor(frameBGR, cv.COLOR_BGR2HSV))

        cameras_frames.append(frame)
        cameras_framesBGR.append(frameBGR)
        cameras_masks.append(BS.compute_mask_in_frame(frame, cam))

    return cameras_masks, cameras_frames, cameras_framesBGR


def handle_frame(videos, cam_numbers, frame_number, prev):
    global C, CM, VR, BS

    cameras_masks, cameras_frames, cameras_framesBGR = determine_cameras_masks_frames(cam_numbers, videos)

    if frame_number == 0:
        voxels = VR.reconstruct_voxels(cameras_masks, None, frame_number)
    else:
        voxels = VR.reconstruct_voxels(cameras_masks, prev, frame_number)

    Assignment.voxels_per_frame.append(voxels)

    voxel_clusters, cluster_centres, compactness = C.cluster(voxels)

    # save_offline_model_information(voxel_clusters,cameras_frames, cameras_framesBGR, frame_number)

    matching = CM.matching_for_frame(voxel_clusters, cameras_frames)  # matching[i] = j if cluster i belongs to model/person j
    matched_cluster_centres = np.zeros((4,2))
    for i in range(len(cluster_centres)):
        matched_cluster_centres[matching[i]] = cluster_centres[i]

    TP.add_to_plot(matched_cluster_centres)

    return cameras_masks


def handle_videos(params):
    global C, CM, VR, BS

    videos = []

    for i in range(params['cam_numbers']):
        videos.append(cv.VideoCapture(os.path.dirname(__file__) + "\\data\\cam" + str(i + 1) + "\\video.avi"))

    prev_cameras_masks = []

    for frame_number in range(params['amount_of_frames']):
        prev_cameras_masks = handle_frame(videos, params['cam_numbers'], frame_number, prev_cameras_masks)

    # add cluster centres with their matching to a list
    # call a plot function which plots the different cluster centres and colours them according to their matching
    Executable.main()


if __name__ == '__main__':
    params = load_parameters()

    init_models(params)

    handle_videos(params)
