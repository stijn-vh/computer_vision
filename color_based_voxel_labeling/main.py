from background_substraction import BackgroundSubstraction
from voxel_reconstruction import VoxelReconstruction
from color_matcher import ColorMatcher
from clustering import Clustering
from trajectory_plotter import TrajectoryPlotter

import numpy as np
import cv2 as cv
from helpers.json_helper import JsonHelper
from helpers.data_generation import DataGenerator


import assignment as Assignment
import executable as Executable

import os

BS = None
VR = None
C = None
CM = None
JH = None
TP = None
DG = None

def load_parameters():
    global JH
    JH = JsonHelper()
    camera_params = JH.load_pickle_object("scaled_camera")
    Assignment.initialise_camera_params(camera_params)
    cam_coords = Assignment.get_cam_positions()
    [max_x, max_y, max_z] = np.max(cam_coords, axis=0).astype(int)
    parameters = {
        'rotation_vectors': [], 'translation_vectors': [], 'intrinsics': [], 'dist_mtx': [],
        'stepsize': 4,
        'max_frame': 100,
        'frame_step':10,
        'cam_numbers': 4,
        'cam_coords': cam_coords,
        'remove_ghosts': False,
        'update_model': True,
        'save_new_offline_model_data' : False,
        'create_new_lookup_table' : False,
        'show_reconstruction': True
    }

    parameters['xb'] = max_x //parameters['stepsize']
    parameters['yb'] = max_y //parameters['stepsize']
    parameters['zb'] = max_z //parameters['stepsize']
    parameters['torso_size'] = 60000 / (parameters['stepsize']** 3) #heuristic


    for camera in camera_params:
        parameters['rotation_vectors'].append(np.array(camera_params[camera]['extrinsic_rvec']))
        parameters['translation_vectors'].append(np.array(camera_params[camera]['extrinsic_tvec']))
        parameters['intrinsics'].append(camera_params[camera]['intrinsic_mtx'])
        parameters['dist_mtx'].append(camera_params[camera]['intrinsic_dist'])

    return parameters


def init_models(params):
    global C, CM, VR, BS, TP, JH, DG

    C = Clustering()
    BS = BackgroundSubstraction()
    BS.create_background_model()
    CM = ColorMatcher(params)
    VR = VoxelReconstruction(params)
    TP = TrajectoryPlotter(params)

    if params['create_new_lookup_table']:
        print('start lookup table creation')
        lookup_table = VR.create_lookup_table()
        print('start lookup table saving to json')
        JH.save_to_json("lookup_table_"+ str(params['stepsize']), lookup_table)
        print('end lookup table saving to json')
    if params['remove_ghosts']:
        VR.create_distance_table()
    if params['save_new_offline_model_data']:
        DG = DataGenerator()

    print('start lookup table loading from json')
    VR.lookup_table = JH.load_from_json('lookup_table_' + str(params['stepsize']))
    print('done lookup table loading from json')

    print("start offline colourmodel creation")
    CM.load_create_offline_model()
    print("end offline colourmodel creation")


def determine_cameras_masks_frames(videos, frame_number):
    # Helper method for handle_frame
    cameras_masks = []
    cameras_frames = []
    cameras_framesBGR = []

    for cam in range(params['cam_numbers']):
        videos[cam].set(cv.CAP_PROP_POS_FRAMES, frame_number)
        ret, frameBGR = videos[cam].read()
        frame = np.float32(cv.cvtColor(frameBGR, cv.COLOR_BGR2HSV))

        cameras_frames.append(frame)
        cameras_framesBGR.append(frameBGR)
        cameras_masks.append(BS.compute_mask_in_frame(frame, cam))

    return cameras_masks, cameras_frames, cameras_framesBGR





def handle_frame(videos, frame_number, prev):
    global C, CM, VR, BS
    print('frame ' + str(frame_number))

    cameras_masks, cameras_frames, cameras_framesBGR = determine_cameras_masks_frames(videos, frame_number)
    if frame_number == 0:
        voxels = VR.reconstruct_voxels(cameras_masks, None, frame_number)
    else:
        voxels = VR.reconstruct_voxels(cameras_masks, prev, frame_number)

    Assignment.voxels_per_frame.append(voxels)

    voxel_clusters, cluster_centres, compactness = C.cluster(voxels)

    if params['save_new_offline_model_data']:
        DG.save_offline_model_information(voxel_clusters,cameras_frames, cameras_framesBGR, frame_number)

    # matching[i] = j if cluster i belongs to model/person j
    matching = CM.matching_for_frame(voxel_clusters, cameras_frames)
    matched_cluster_centres = CM.match_cluster_centres(cluster_centres, matching)
    matched_clusters = CM.match_clusters(voxel_clusters, matching)

    #TP.add_to_plot(matched_cluster_centres)
    print(matched_cluster_centres)

    return cameras_masks


def handle_videos(params):
    global C, CM, VR, BS

    videos = []

    for i in range(params['cam_numbers']):
        videos.append(cv.VideoCapture(os.path.dirname(__file__) + "\\data\\cam" + str(i + 1) + "\\video.avi"))

    prev_cameras_masks = []

    for frame_number in range(0, params['max_frame'], params['frame_step']):
        prev_cameras_masks = handle_frame(videos, frame_number, prev_cameras_masks)

    if params['show_reconstruction']:
        Executable.main()


if __name__ == '__main__':
    params = load_parameters()

    init_models(params)

    handle_videos(params)
