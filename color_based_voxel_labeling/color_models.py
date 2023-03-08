import numpy as np
import cv2 as cv
import sklearn as sk


class colour_models:
    rotation_vectors = []
    translation_vectors = []
    intrinsics = []
    dist_mtx = []
    cam_offline_color_models = []


    # single camera
    def voxels_to_colors(self, voxel_clusters, frames, cam):
        # Every element in array is a list of [x,y,z] voxels corresponding to a persons upper body
        #for example: voxel_clusters = [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9]], [[10, 11, 12]], [[13, 14, 15], [16, 17, 18]]]
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
            color_clusters.append(frames[cam][iy][ix])
        return color_clusters  # similar shape as voxel_clusters, where now a list of HSVs values is given for every voxel cluster

    # single camera
    def create_offline_model(self, cam_voxel_clusters, frames):
        #cam_voxel_clusters contains for every camera a "voxel_clusters" which need to contain seperated clusters.
        # Furthermore, we require that the 4 clusters in each voxel_clusters are ordered the same for every camera,
        # so that all i-th clusters belong to the same person.
        offline_color_model = []
        for cam in range(4):
            color_clusters = self.voxels_to_colors(cam_voxel_clusters[cam], frames, cam)
            for cluster in range(4):
                gmm = sk.mixture.GaussianMixture(n_components=1, random_state=0).fit(color_clusters[cluster])
                offline_color_model.append(gmm)
            self.cam_offline_color_models.append(offline_color_model)

    # single camera
    def color_model_scores(self, color_clusters, offline_color_model):
        scores = np.zeros((4, 4))
        for model in range(4):
            for cluster in range(4):
                scores[model][cluster] = offline_color_model[model].score(color_clusters[cluster])
        return scores

    def run_matching_for_frame(self, voxel_clusters, frames):
        #Assumes that self.cam_offline_color_models has been created already
        #self.cam_offline_color_models: for every camera, contains a GMM for each of the 4 persons.

        # voxel_clusters contains 4 clusters, one for each person in a specific frame
        # frames shape: (4, 486, 644, 3) corresponding to cameras, pixel coordinates(y,x), HSV  in a specific frame
        total_scores = np.zeros((4, 4))
        for cam in range(4):
            color_clusters = self.voxels_to_colors(voxel_clusters, frames, cam)
            total_scores += self.color_model_scores(color_clusters, self.cam_offline_color_models[cam])
        return self.hungarian_matching(total_scores)



    def hungarian_matching(self, scores):
        #Use the Hungarian algorithm to find an optimal matching between the models (rows in scores) and the current clusters (columns in scores)
        matching = np.zeros((4,4)) #matchin[i][j] = 1 if cluster j belongs to model i
        return matching




