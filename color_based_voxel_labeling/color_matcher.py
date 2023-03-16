import numpy as np
import cv2 as cv
from scipy.optimize import linear_sum_assignment
from offline_color_model import OfflineColorModel

class ColorMatcher(OfflineColorModel):
    def approximate_voxel_cluster(self, x_centre, z_centre):
        voxel_rect_around_centre = []
        radius = 10 // self.stepsize
        height = 60 // self.stepsize
        for x in range(-radius, radius):
            for z in range(-radius, radius):
                for y in range(height):
                    voxel_rect_around_centre.append(
                        [x_centre + self.stepsize * x, 100 + self.stepsize * y, z_centre + self.stepsize * z])
        return np.array(voxel_rect_around_centre, dtype=int)

    def update_voxel_clusters(self, voxel_clusters):
        for cluster in range(len(voxel_clusters)):
            if len(voxel_clusters[cluster]) < self.torso_size/2:
                centres = np.sum(voxel_clusters[cluster], axis=0) / len(voxel_clusters[cluster])
                x_centre = centres[0]
                z_centre = centres[2]
                print("creating approximate voxel cluster with approximate centre", x_centre, z_centre)
                np.append(voxel_clusters[cluster], self.approximate_voxel_cluster(x_centre, z_centre), axis=0)
                self.update_model = False
    def color_model_scores(self, color_clusters, cam):
        scores = np.zeros((4, 4))
        for model in range(4):
            for cluster in range(4):
                scores[cluster][model] = self.cam_offline_color_models[cam][model].score(color_clusters[cluster])
        return scores
    def hungarian_matching(self, scores):
        # Use the Hungarian algorithm to find an optimal matching between current clusters (rows in scores) and the models (columns in scores)
        row_ind, col_ind = linear_sum_assignment(-scores)
        return col_ind

    def match_cluster_centres(self, cluster_centres, matching):
        matched_cluster_centres = np.zeros((4, 2))
        for i in range(len(cluster_centres)):
            matched_cluster_centres[matching[i]] = cluster_centres[i]
        return matched_cluster_centres

    def match_clusters(self,  clusters, matching):
        matched_clusters =[]
        for model in range(len(matching)):
            matched_clusters.append(clusters[np.where(matching == model)[0][0]])
        return matched_clusters

    def matching_for_frame(self, voxel_clusters, cameras_frames):
        # Assumes that self.cam_offline_color_models has been created already,
        # which for every camera, contains a GMM for each of the 4 persons.

        self.update_voxel_clusters(voxel_clusters) #If a cluster is too small create an approximate one
        cam_color_clusters = []
        total_scores = np.zeros((4, 4))
        for cam in range(len(cameras_frames)):
            color_clusters = self.clusters_to_colors(voxel_clusters, cameras_frames[cam], cam, offline=False)
            cam_color_clusters.append(color_clusters)
            total_scores += self.color_model_scores(color_clusters, cam)
        matching = self.hungarian_matching(total_scores)  # matching[i] = j if cluster i belongs to model/person j

        if self.update_model:
            print("updating colormodel")
            for cam in range(len(cameras_frames)):
                for model in range(len(matching)):
                    self.cam_offline_color_models[cam][model].fit(
                        cam_color_clusters[cam][np.where(matching == model)[0][0]])
        self.update_model = True  # If it was set to false with an approximate cluster
        return matching

