import cv2 as cv
from sklearn.mixture import GaussianMixture
from helpers.json_helper import JsonHelper
import numpy as np
from color_model import ColorModel


class OfflineColorModel(ColorModel):
    #Base class for the ColorMatcher
    def plot_projected_voxels(self, voxel_clusters, frame, cam):
        # Helper method used to understand which voxel cluster belongs to which real life person
        for person in range(len(voxel_clusters)):
            ix, iy = self.voxels_to_indices(voxel_clusters[person], cam)
            if person == 0:
                frame[(iy, ix)] = [255, 255, 255]
            elif person == 1:
                frame[(iy, ix)] = [255, 255, 0]
            elif person == 2:
                frame[(iy, ix)] = [0, 255, 255]
            elif person == 3:
                frame[(iy, ix)] = [255, 0, 255]
        cv.imshow("frame cam " + str(cam), frame)
        cv.waitKey(10000)
    def create_offline_model(self, four_good_offline_voxel_clusters_per_camera, corresponding_frame_per_camera):
        # We require that the 4 clusters are ordered the same for every camera,
        # so that all i-th clusters belong to the same person.
        cam_offline_color_models = []
        for cam in range(4):
            offline_color_model = []
            color_clusters = self.clusters_to_colors(four_good_offline_voxel_clusters_per_camera[cam],
                                                     corresponding_frame_per_camera[cam], cam, offline= True)
            for cluster in range(4):
                gmm = GaussianMixture(n_components=3, random_state=0, max_iter=1000, tol=1e-10)
                gmm.fit(color_clusters[cluster])
                # For future model updates:
                gmm.warm_start = True
                gmm.tol = self.update_tolerance
                offline_color_model.append(gmm)
            cam_offline_color_models.append(offline_color_model)
        return cam_offline_color_models

    def load_create_offline_model(self):
        # Loads clusters from selected frames for which it is known that all 4 clusters are well seperated
        # and for which it is also known (manually checked) which voxel cluster belongs to which real life person.
        JH = JsonHelper()
        clusters124 = JH.load_from_json("clusters_cam_1_2_4")
        camframes124 = np.array(JH.load_from_json("cameras_frames_1_2_4"))
        clusters3 = JH.load_from_json("clusters_cam_3")
        camframes3 = np.array(JH.load_from_json("cameras_frames_3"))
        rearranged_cluster3 = [np.array(clusters3[1]), np.array(clusters3[2]), np.array(clusters3[0]),
                               np.array(clusters3[3])]
        np_clusters124 = []
        for i in range(len(clusters124)):
            np_clusters124.append(np.array(clusters124[i]))
        four_good_offline_voxel_clusters_per_camera = [np_clusters124, np_clusters124, rearranged_cluster3,
                                                       np_clusters124]

        corresponding_frame_per_camera = np.array([camframes124[0], camframes124[1], camframes3[2], camframes124[3]])
        self.cam_offline_color_models = self.create_offline_model(four_good_offline_voxel_clusters_per_camera, corresponding_frame_per_camera)