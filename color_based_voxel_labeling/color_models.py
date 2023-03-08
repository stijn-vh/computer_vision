import numpy as np
import cv2 as cv
import sklearn as sk


class colour_models:
    #Every element in array is a list of [x,y,z] voxels corresponding to a persons upper body
    voxel_clusters = [[[1,2,3], [4,5,6]] , [[7,8,9]], [[10,11,12]], [[13,14,15],[16,17,18]]]
    frames = [] # frames shape: (4, 428, 486, 644, 3). corresponding to cameras, frames, pixel coordinates(y,x), HSV
    rotation_vectors = []
    translation_vectors = []
    intrinsics = []
    dist_mtx = []


    def voxels_to_colors(self, voxel_clusters, cam, frame_num):
        color_clusters = []
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
            #This most likely doesnt work yet.
            #Want colours to
            color_clusters.append(self.frames[cam][frame_num][iy][ix])
        return color_clusters #similar shape as voxel_clusters, where now a list of HSVs values is given for every voxel cluster

    def create_offline_model(self, color_clusters):
        offline_color_models = []
        for cluster in range(4):
            gmm = sk.mixture.GaussianMixture(n_components=1, random_state=0).fit(color_clusters[cluster])
            offline_color_models.append(gmm)

