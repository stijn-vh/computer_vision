import glm
import random
import numpy as np
import pickle

block_size = 1.0

# TODO create method which reads in these values and assigns them from task 1 and task 3.
rotation_matrices = []  # needs to be filled
translation_vectors = []  # needs to be filled
voxels = []  # Need te be filled from voxel reconstruction


def load_parameters_from_pickle(path):
    with open(path, 'rb') as f:
        camera_params = pickle.load(f)

        for camera in camera_params:
            rotation_matrices.append(camera_params[camera]['R'])
            rotation_matrices.append(camera_params[camera]['extrinsic_tvec'])

def generate_grid(width, depth):
    # Generates the floor grid locations
    # You don't need to edit this function
    data = []
    for x in range(width):
        for z in range(depth):
            data.append([x * block_size - width / 2, -block_size, z * block_size - depth / 2])
    return data


def set_voxel_positions(width, height, depth):
    # TODO: You need to calculate proper voxel arrays instead of random ones.
    return voxels

    # # Generates random voxel locations
    # # You can set the voxel locations using list of lists. [[x1, y2, z1], [x2, y2, z2], ..., [xn, yn, zn]].
    # data = []
    # for x in range(width):
    #     for y in range(height):
    #         for z in range(depth):
    #             if random.randint(0, 1000) < 5:
    #                 data.append([x * block_size - width / 2, y * block_size, z * block_size - depth / 2])
    # return data


"""
    https://stackoverflow.com/questions/18637494/camera-position-in-world-coordinate-from-cvsolvepnp
    Given the rotation matrix R and translation vector t which translate world to camera coordinates, we want to find the inverse
    which translates camera to world coordinates. Then we translate the origin of the camera to world coordinates.
    The inverse of y= Rx +t is given by x = R.Ty - R.T*t as explained in the link. 
    Now with y the origin we get that R.Ty =0 so - R.T*t gives the camera coordinates
"""


def get_cam_positions():
    camera_coords = []
    for i in range(4):
        camera_coords.append(- rotation_matrices[i].T @ translation_vectors[i])
    return camera_coords
    # return [[-64 * block_size, 64 * block_size, 63 * block_size],
    #         [63 * block_size, 64 * block_size, 63 * block_size],
    #         [63 * block_size, 64 * block_size, -64 * block_size],
    #         [-64 * block_size, 64 * block_size, -64 * block_size]]


def get_cam_rotation_matrices():
    # TODO: You need to input the estimated camera rotation matrices (4x4) of the 4 cameras in the world coordinates.
    cam_rotations = []
    rotation_matrix = np.eye(4)
    for i in range(4):
        rotation_matrix[0:3, 0:3] = rotation_matrices[i]
        rotation_matrix[0:3, 3] = translation_vectors[i]
        cam_rotations.append(glm.make_mat4(rotation_matrix))
    return cam_rotations
    # Generates dummy camera rotation matrices, looking down 45 degrees towards the center of the room
    # cam_angles = [[0, 45, -45], [0, 135, -45], [0, 225, -45], [0, 315, -45]]
    # cam_rotations = [glm.mat4(1), glm.mat4(1), glm.mat4(1), glm.mat4(1)]
    # for c in range(len(cam_rotations)):
    #     cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][0] * np.pi / 180, [1, 0, 0])
    #     cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][1] * np.pi / 180, [0, 1, 0])
    #     cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][2] * np.pi / 180, [0, 0, 1])
    # return cam_rotations

