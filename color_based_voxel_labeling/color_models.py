import numpy as np
import cv2 as cv
import sklearn as sk


class colour_models:
    #Every element in array is a list of [x,y,z] voxels corresponding to a persons upper body
    voxel_clusters = [[[1,2,3], [4,5,6]] , [[7,8,9]], [[10,11,12]], [[13,14,15],[16,17,18]]]
    frames = [] # frames shape: (4, 428, 3, 486, 644). corresponding to cameras, frames, HSV, pixel coordinates
    rotation_vectors = []
    translation_vectors = []
    intrinsics = []
    dist_mtx = []

    offline_color_models = []
    def get_pixel_colors(self, voxel_clusters, cam, frame_num):
        for person in range(4):
            idx = cv.projectPoints(
                voxel_clusters[person],
                self.rotation_vectors[cam],
                self.translation_vectors[cam],
                self.intrinsics[cam],
                distCoeffs=self.dist_mtx[cam]
            )
            ix = idx[0][:, 0][:, 0]
            iy = idx[0][:, 0][:, 1]
            #self.frames[cam][frame_num]



        return