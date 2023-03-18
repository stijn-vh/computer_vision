import numpy as np
import cv2 as cv
from voxel_model import VoxelModel

class ColorModel(VoxelModel):
    #Base class for the OfflineColorModel
    rotation_vectors = []
    translation_vectors = []
    intrinsics = []
    dist_mtx = []
    cam_offline_color_models = []
    inv_lookup_table = []

    def __init__(self, params) -> None:
        super().__init__(params)
        self.update_model = params['update_model']
        self.torso_size = params['torso_size']
        self.update_tolerance = params['update_tolerance']
        
    def clusters_to_colors(self, voxel_clusters, frame, cam, offline=False):
        #Translates 4 clusters of voxels to 4 clusters of HSVs
        color_clusters = []
        for person in range(len(voxel_clusters)):
            ix, iy = self.project_voxels(voxel_clusters[person], cam)
            if offline:
                ix,iy, _ = self.get_pixels_in_range(ix,iy)
            color_clusters.append(frame[(iy, ix)])
        return color_clusters