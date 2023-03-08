import cv2 as cv

class Clustering:
    criteria = None
    flags = None
    K = None

    def __init__(self) -> None:
       self.criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
       self.flags = cv.KMEANS_RANDOM_CENTERS
       self.K = 4
    
    def cluster(self, voxels):
        # Transform voxels from [x, y, z] to [x, y]
        data = voxels[:, 0:1]
        compactness, labels, center = cv.kmeans(data, self.K, None, self.criteria, 10, self.flags)
        
        return 


