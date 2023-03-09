import numpy as np
import cv2 as cv
import sklearn as sk
import pickle
from scipy.optimize import linear_sum_assignment


class ColourModels:
    rotation_vectors = []
    translation_vectors = []
    intrinsics = []
    dist_mtx = []
    cam_offline_color_models = []

    def __init__(self, params) -> None:
        self.rotation_vectors = params['rotation_vectors']
        self.translation_vectors = params['translation_vectors']
        self.intrinsics = params['intrinsics']
        self.dist_mtx = params['dist_mtx']
        self.stepsize = params['stepsize']

    # single camera
    def voxels_to_colors(self, voxel_clusters, frame, cam):
        color_clusters = []
        for person in range(4):
            # For projecting the voxels, we need to switch the x,y,z visualisation coordinates to normal coordinates (x, z, -y)
            temp = np.copy(voxel_clusters[person][:, 2])
            voxel_clusters[person][:, 2] = -voxel_clusters[person][:, 1]
            voxel_clusters[person][:, 1] = temp
            idx = cv.projectPoints(
                voxel_clusters[person],
                self.rotation_vectors[cam],
                self.translation_vectors[cam],
                self.intrinsics[cam],
                distCoeffs=self.dist_mtx[cam]
            )
            ix = idx[0][:, 0][:, 0]
            iy = idx[0][:, 0][:, 1]
            # This most likely doesnt work yet.
            # Want colours to
            color_clusters.append(frame[iy][ix])
        return color_clusters  # similar shape as voxel_clusters, where now a list of HSVs values is given for every voxel cluster

    # single camera
    def create_offline_model(self, four_good_offline_voxel_clusters_per_camera, corresponding_frame_per_camera):
        # We require that the 4 clusters are ordered the same for every camera,
        # so that all i-th clusters belong to the same person.
        offline_color_model = []
        for cam in range(4):
            color_clusters = self.voxels_to_colors(four_good_offline_voxel_clusters_per_camera[cam],
                                                   corresponding_frame_per_camera[cam])
            for cluster in range(4):
                gmm = sk.mixture.GaussianMixture(n_components=1, random_state=0).fit(color_clusters[cluster])
                offline_color_model.append(gmm)
            self.cam_offline_color_models.append(offline_color_model)

    # single camera
    def color_model_scores(self, offline_color_model,  color_clusters):
        scores = np.zeros((4, 4))
        for model in range(4):
            for cluster in range(4):
                scores[cluster][model] = offline_color_model[model].score(color_clusters[cluster])
        return scores

    def create_approximate_voxel_cluster(self,x_centre, z_centre):
        print("creating approximate voxel cluster")
        voxel_rect_around_centre = [] 
        radius = 15 // self.stepsize
        height = 80 // self.stepsize
        for x in self.stepsize*range(-radius, radius):
            for z in self.stepsize*range(-radius, radius):
                for y in self.stepsize*range(height):
                    voxel_rect_around_centre.append([x_centre +x ,100+height, z_centre+ z])
        return voxel_rect_around_centre



    def matching_for_frame(self, voxel_clusters,   cameras_frames):
        # Assumes that self.cam_offline_color_models has been created already,
        # which for every camera, contains a GMM for each of the 4 persons.
        for cluster in range(len(voxel_clusters)):
            if len(voxel_clusters[cluster]) < 500:
                #For when the upper body is missing,
                # less than ... voxels in cluster with y value between 100 and 180cm
                centres = np.sum(voxel_clusters[cluster], axis =0)/len(voxel_clusters[cluster])
                x_centre = centres[0]
                z_centre = centres[2]
                voxel_clusters[cluster] = self.create_approximate_voxel_cluster(x_centre, z_centre)

        total_scores = np.zeros((4, 4))
        for cam in range(len(cameras_frames)):
            color_clusters = self.voxels_to_colors(voxel_clusters,  cameras_frames[cam], cam)
            total_scores += self.color_model_scores(self.cam_offline_color_models[cam], color_clusters)
        return self.hungarian_matching(total_scores)

    def hungarian_matching(self, scores):
        # Use the Hungarian algorithm to find an optimal matching between current clusters (rows in scores) and the models (columns in scores)
        row_ind, col_ind = linear_sum_assignment(scores)
        return col_ind
