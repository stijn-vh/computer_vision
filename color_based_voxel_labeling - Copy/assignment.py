import glm
import random
import numpy as np
from helpers.json_helper import JsonHelper

JH = JsonHelper()
camera_params = JH.load_pickle_object("scaled_camera")

block_size = 1.0
voxels_per_frame = []  # Need te be filled from voxel reconstruction
cluster_sizes = []
frame = 0

def get_camera_params():
    rotation_matrices = []
    translation_vectors = []
    for camera in camera_params:
        rotation_matrices.append(camera_params[camera]['R'][0])
        translation_vectors.append(camera_params[camera]['extrinsic_tvec'].reshape(-1))
    return rotation_matrices, translation_vectors


def generate_grid(width, depth):
    # Generates the floor grid locations
    # You don't need to edit this function
    data, colors = [], []
    for x in range(width):
        for z in range(depth):
            data.append([x*block_size - width/2, -block_size, z*block_size - depth/2])
            colors.append([1.0, 1.0, 1.0] if (x+z) % 2 == 0 else [0, 0, 0])
    return data, colors


def set_voxel_positions(width, height, depth):
    global frame
    colors = np.empty([1, 3])

    values = [[0, 0, 90], [90, 0, 0], [90, 40, 0], [0, 90, 0]]
    voxels_to_return = voxels_per_frame[frame]

    for i in range(len(cluster_sizes[frame])):
        x = np.full((cluster_sizes[frame][i], len(values[i])), values[i])
        colors = np.concatenate((colors, x))

    if frame == len(voxels_per_frame) - 1:
        frame = 0
    else:
        frame += 1

    return voxels_to_return, colors[1:]

def get_cam_positions():
    rotation_matrices, translation_vectors = get_camera_params()
    camera_coords = []
    camera_colors = [150, 250, 330, 460]
    for i in range(4):
        [x, y, z] = - rotation_matrices[i].T @ translation_vectors[i]
        camera_coords.append([x, -z, y])
    return np.array(camera_coords), np.array(camera_colors)

# def get_cam_rotation_matrices():
#     cam_angles = [[0, 45, -45], [0, 135, -45], [0, 225, -45], [0, 315, -45]]
#     cam_rotations = [glm.mat4(1), glm.mat4(1), glm.mat4(1), glm.mat4(1)]
#     for c in range(len(cam_rotations)):
#         cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][0] * np.pi / 180, [1, 0, 0])
#         cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][1] * np.pi / 180, [0, 1, 0])
#         cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][2] * np.pi / 180, [0, 0, 1])
#     return cam_rotations

def get_cam_rotation_matrices():
    rotation_matrices, translation_vectors = get_camera_params()
    cam_rotations = []
    for cam in range(4):
        glm_mat = glm.mat4([[rotation_matrices[cam][0][0], rotation_matrices[cam][0][2], rotation_matrices[cam][0][1],
                             translation_vectors[cam][0]],
                            [rotation_matrices[cam][1][0], rotation_matrices[cam][1][2], rotation_matrices[cam][1][1],
                             translation_vectors[cam][1]],
                            [rotation_matrices[cam][2][0], rotation_matrices[cam][2][2], rotation_matrices[cam][2][1],
                             translation_vectors[cam][2]],
                            [0, 0, 0, 1]])
        glm_mat = glm.rotate(glm_mat, glm.radians(90), (0, 1, 1))
        cam_rotations.append(glm_mat)
    return cam_rotations