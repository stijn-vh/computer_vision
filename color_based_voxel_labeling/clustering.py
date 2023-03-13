import cv2 as cv
import numpy as np
from scipy import spatial

class Clustering:
    criteria = None
    flags = None
    K = None

    old_centres = []
    new_matched_indices = []

    def __init__(self) -> None:
       self.criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
       self.flags = cv.KMEANS_RANDOM_CENTERS
       self.K = 4
    
    def cluster(self, voxels):
        # Transform voxels from [x, y, z] to [x, y]
        data = np.float32(voxels[:, [0,2]])
        compactness, labels, centers = cv.kmeans(data, self.K, None, self.criteria, 10, self.flags)
        
        voxel_clusters = []

        for label in enumerate(np.unique(labels)):
            # Take z where 100 < z < 180 
            label_idx = np.where(labels == label[0])[0]

            voxels_with_label = np.take(voxels, label_idx, axis = 0)

            z_idx = np.where(np.logical_and(voxels_with_label[:, 1] > 100, voxels_with_label[:, 1] < 180))
            voxel_clusters.append(np.take(voxels_with_label, z_idx, axis = 0)[0])

        centers = self.matching_based_on_centres(centers)
        return voxel_clusters, centers, compactness
    
    def find_closest_point_index(self, point):
        distances = np.linalg.norm(self.old_centres - point, axis=1)
        min_index = np.argmin(distances)

        return min_index


    def matching_based_on_centres(self, centres):
        centres = np.array(centres)
        new_centres = np.zeros((4, 2))

        if (len(self.old_centres) == 0):
            self.old_centres = centres
            new_centres = centres
        else:
            for centre in centres:
                new_centres[self.find_closest_point_index(centre)] = centre

            self.old_centres = new_centres
        return new_centres


