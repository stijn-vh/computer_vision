import glm
import random
import numpy as np
import pickle

block_size = 1

# TODO create method which reads in these values and assigns them from task 1 and task 3.
rotation_matrices = []  # needs to be filled
translation_vectors = []  # needs to be filled
voxels = []  # Need te be filled from voxel reconstruction


def load_parameters_from_pickle(path):
    with open(path, 'rb') as f:
        camera_params = pickle.load(f)

        for camera in camera_params:
            rotation_matrices.append(camera_params[camera]['R'][0])
            translation_vectors.append(camera_params[camera]['extrinsic_tvec'].reshape(-1))


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
        [x, y, z] = - rotation_matrices[i].T @ translation_vectors[i]
        camera_coords.append([x, -z, y])  # check if this is the correct transformation in camera coords
    return camera_coords


def get_cam_rotation_matrices():
    # TODO: You need to input the estimated camera rotation matrices (4x4) of the 4 cameras in the world coordinates.
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
