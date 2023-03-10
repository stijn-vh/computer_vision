import numpy as np
import cv2 as cv
import copy
from sklearn.mixture import GaussianMixture
from json_helper import JsonHelper
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

    #Possibly refactor code duplication
    def plot_projected_voxels(self, voxel_clusters, frame, cam):
        voxel_clusters1 = copy.deepcopy(voxel_clusters)
        for person in range(len(voxel_clusters1)):
            # For projecting the voxels, we need to switch the x,y,z visualisation coordinates to normal coordinates (x, z, -y)
            temp = np.copy(voxel_clusters1[person][:, 2])
            voxel_clusters1[person][:, 2] = -voxel_clusters1[person][:, 1]
            voxel_clusters1[person][:, 1] = temp
            idx = cv.projectPoints(
                np.float32(voxel_clusters1[person]),
                self.rotation_vectors[cam],
                self.translation_vectors[cam],
                self.intrinsics[cam],
                distCoeffs=self.dist_mtx[cam]
            )
            ix = idx[0][:, 0][:, 0].astype(int)
            iy = idx[0][:, 0][:, 1].astype(int)

            if person == 0:
                frame[(iy, ix)] = [255, 255, 255]
            elif person == 1:
                frame[(iy, ix)] = [255, 255, 0]
            elif person == 2:
                frame[(iy, ix)] = [0, 255, 255]
            elif person == 3:
                frame[(iy, ix)] = [255, 0, 255]
        cv.imshow("frame cam "+str(cam), frame)
        cv.waitKey(10000)

    # single camera
    def voxels_to_colors(self, voxel_clusters, frame, cam):
        voxel_clusters1 = copy.deepcopy(voxel_clusters) #otherwise weird pointer behaviour
        color_clusters = []
        for person in range(len(voxel_clusters1)):
            # For projecting the voxels, we need to switch the x,y,z visualisation coordinates to normal coordinates (x, z, -y)
            temp = copy.deepcopy(voxel_clusters1[person][:, 2])
            voxel_clusters1[person][:, 2] = -voxel_clusters1[person][:, 1]
            voxel_clusters1[person][:, 1] = temp
            idx = cv.projectPoints(
                np.float32(voxel_clusters1[person]),
                self.rotation_vectors[cam],
                self.translation_vectors[cam],
                self.intrinsics[cam],
                distCoeffs=self.dist_mtx[cam]
            )
            ix1 = idx[0][:, 0][:, 0].astype(int)
            iy1 = idx[0][:, 0][:, 1].astype(int)
            iiy = np.asarray(abs(iy1) < 486).nonzero()
            iix = np.asarray(abs(ix1) < 644).nonzero()

            indices = np.intersect1d(iix, iiy)

            ix = np.take(ix1, indices).astype(int)
            iy = np.take(iy1, indices).astype(int)

            if not np.array_equiv(ix, ix1):
                print("?")
            if not np.array_equiv(iy, iy1):
                print("?")

            color_clusters.append(frame[(iy, ix)])
        return color_clusters  # similar shape as voxel_clusters, where now a list of HSVs values is given for every voxel cluster

    def create_offline_model(self, four_good_offline_voxel_clusters_per_camera, corresponding_frame_per_camera):
        # We require that the 4 clusters are ordered the same for every camera,
        # so that all i-th clusters belong to the same person.
        for cam in range(4):
            offline_color_model = []
            color_clusters = self.voxels_to_colors(four_good_offline_voxel_clusters_per_camera[cam],
                                                   corresponding_frame_per_camera[cam], cam)
            for cluster in range(4):
                gmm = GaussianMixture(n_components=1, random_state=0).fit(color_clusters[cluster])
                offline_color_model.append(gmm)
            self.cam_offline_color_models.append(offline_color_model)

    def load_create_offline_model(self):
        JH = JsonHelper()
        clusters124 = JH.load_from_json("clusters_cam_1_2_4")
        camframes124 =  np.array(JH.load_from_json("cameras_frames_1_2_4"))
        clusters3 =  JH.load_from_json("clusters_cam_3")
        camframes3 =  np.array(JH.load_from_json("cameras_frames_3"))
        rearranged_cluster3 =  [np.array(clusters3[1]), np.array(clusters3[2]), np.array(clusters3[0]), np.array(clusters3[3])]
        np_clusters124 = []
        for i in range(len(clusters124)):
            np_clusters124.append(np.array(clusters124[i]))
        four_good_offline_voxel_clusters_per_camera = [np_clusters124, np_clusters124, rearranged_cluster3, np_clusters124]

        corresponding_frame_per_camera = np.array([camframes124[0], camframes124[1], camframes3[2], camframes124[3]])
        self.create_offline_model(four_good_offline_voxel_clusters_per_camera, corresponding_frame_per_camera)

    # single camera
    def color_model_scores(self, offline_color_model, color_clusters):
        scores = np.zeros((4, 4))
        for model in range(4):
            for cluster in range(4):
                scores[cluster][model] = offline_color_model[model].score(color_clusters[cluster])
        return scores

    def create_approximate_voxel_cluster(self, x_centre, z_centre):
        print("creating approximate voxel cluster")
        voxel_rect_around_centre = []
        radius = 15 // self.stepsize
        height = 80 // self.stepsize
        for x in range(-radius, radius):
            for z in range(-radius, radius):
                for y in range(height):
                    voxel_rect_around_centre.append(
                        [x_centre + self.stepsize * x, 100 + self.stepsize * y, z_centre + self.stepsize * z])
        return voxel_rect_around_centre

    def matching_for_frame(self, voxel_clusters, cameras_frames):
        # Assumes that self.cam_offline_color_models has been created already,
        # which for every camera, contains a GMM for each of the 4 persons.
        for cluster in range(len(voxel_clusters)):
            if len(voxel_clusters[cluster]) < 30000 / (self.stepsize ** 3):
                # For when the upper body is missing,
                # less than ... voxels in cluster with y value between 100 and 180cm
                centres = np.sum(voxel_clusters[cluster], axis=0) / len(voxel_clusters[cluster])
                x_centre = centres[0]
                z_centre = centres[2]
                voxel_clusters[cluster] = self.create_approximate_voxel_cluster(x_centre, z_centre)

        total_scores = np.zeros((4, 4))
        for cam in range(len(cameras_frames)):
            color_clusters = self.voxels_to_colors(voxel_clusters, cameras_frames[cam], cam)
            total_scores += self.color_model_scores(self.cam_offline_color_models[cam], color_clusters)
        return self.hungarian_matching(total_scores)

    def hungarian_matching(self, scores):
        # Use the Hungarian algorithm to find an optimal matching between current clusters (rows in scores) and the models (columns in scores)
        row_ind, col_ind = linear_sum_assignment(scores)
        return col_ind
