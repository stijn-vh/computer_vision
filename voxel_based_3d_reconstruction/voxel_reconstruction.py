import numpy as np

class VoxelReconstruction:
    #TODO method which reads in these values
    rotation_vectors = []
    translation_vectors =[]
    intrinsics = []
    def __init__(self, ) -> None:
        pass
    def create_lookup_table(self, masks):
        #masks shape: (4, 428, 486, 644)
        #for every camera, for every frame in camera, a mask with white pixels in foreground and black pixels in background
        #Lookup table will contain for every camera, for every pixel value, a list of voxel coords which are projected to that pixel value
        lookup_table = np.zeros((4,486,644)) #needs to have a list at every elem A=np.array([[1,2],[],[1,2,3,4]])
        