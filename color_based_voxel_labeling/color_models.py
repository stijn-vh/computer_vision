import numpy as np
import cv2 as cv
import sklearn as sk
import pickle

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

    # single camera
    def voxels_to_colors(self, voxel_clusters, frame, cam):
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
                scores[model][cluster] = offline_color_model[model].score(color_clusters[cluster])
        return scores

    def matching_for_frame(self, voxel_clusters,   cameras_frames):
        # Assumes that self.cam_offline_color_models has been created already,
        # which for every camera, contains a GMM for each of the 4 persons.
        total_scores = np.zeros((4, 4))
        for cam in range(4):
            color_clusters = self.voxels_to_colors(voxel_clusters,  cameras_frames[cam], cam)
            total_scores += self.color_model_scores(self.cam_offline_color_models[cam], color_clusters)
        return self.hungarian_matching(total_scores)

    def hungarian_matching(self, scores):
        # Use the Hungarian algorithm to find an optimal matching between the models (rows in scores) and the current clusters (columns in scores)
        matching = np.zeros((4, 4))
        return matching
