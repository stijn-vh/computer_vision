from calibration import Calibration
from color_based_voxel_labeling.helpers.json_helper import JsonHelper
from background_substraction import BackgroundSubstraction
import numpy as np

class DataGenerator:
    def __init__(self,CM) -> None:
        self.JH = JsonHelper()
        self.CM = CM

    def determine_camera_params(self):
        cali = Calibration()
        cali.obtain_intrinsics_from_cameras()
        cali.obtain_extrinsics_from_cameras()
        self.JH.pickle_object("scaled_camera", cali.cameras)


    def determine_new_masks(self,show_video=True):
        S = BackgroundSubstraction()
        cam_means, cam_std_devs = S.create_background_model()
        thresholds = np.array([[10, 2, 18],
                               [10, 2, 14],
                               [10, 1, 14],
                               [10, 2, 20]])
        num_contours = [4, 5, 5, 6]
        masks, frames = S.background_subtraction(thresholds, num_contours, cam_means, cam_std_devs, show_video)
        return masks, frames


    def determine_new_thresholds(self):
        S = BackgroundSubstraction()
        cam_means, cam_std_devs = S.create_background_model()
        thresh, num_contours = S.gridsearch(cam_means, cam_std_devs)
        print("thresholds = ", thresh, "Num_contours =", num_contours)


    def save_offline_model_information(self, voxel_clusters, cameras_frames, cameras_framesBGR, frame_number):
        if frame_number == 0:
            self.JH.save_to_json("clusters_cam_1_2_4", voxel_clusters)
            self.JH.save_to_json("cameras_frames_1_2_4", cameras_frames)
            self.CM.plot_projected_voxels(voxel_clusters, cameras_framesBGR[0], 0)
            self.CM.plot_projected_voxels(voxel_clusters, cameras_framesBGR[1], 1)
            self.CM.plot_projected_voxels(voxel_clusters, cameras_framesBGR[3], 3)
            print("done frame 0")
        if frame_number == 180:
            self.JH.save_to_json("clusters_cam_3", voxel_clusters)
            self.JH.save_to_json("cameras_frames_3", cameras_frames)
            self.CM.plot_projected_voxels(voxel_clusters, cameras_framesBGR[2], 2)
            print("done frame 180")