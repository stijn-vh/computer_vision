import numpy as np
import cv2 as cv


class VoxelReconstruction:
    # TODO method which reads in these values
    rotation_vectors = []
    translation_vectors = []
    intrinsics = []

    def __init__(self, ) -> None:
        pass

    def create_lookup_table(self):
        # The lookup_table shape is (4,486,644) for indexing the cameras and the pixels of each camera.
        # Each pixels stores a variable sized list of multiple [x,y,z] coordinates.

        # un-hardcode these values
        lookup_table = [[[[] for _ in range(644)] for _ in range(486)] for _ in range(4)]
        all_voxels = [[x,y,z] for x in range(128) for y in range(128) for z in range(64)]
        for cam in range(4):
            for vox in all_voxels:
                [ix, iy] = cv.projectPoints(vox, self.rotation_vectors[cam],
                                            self.translation_vectors[cam],
                                            self.intrinsics[cam])
                lookup_table[cam][ix][iy].append(vox)
        return lookup_table

    def return_visible_voxels(self, mask, cam_lookup_table):
        # mask has shape (486,644) corresponding to the pixels in a single frame
        # Possibly vectorize this and/or decrease time complexity by comparing the current mask with the previous mask.
        # Maybe this could be done so that only the changed pixels (XOR?) add or remove voxels from a stored list of voxels.
        vis_vox = []
        for ix in range(len(mask)):
            for iy in range(len(mask[0])):
                if mask[ix][iy] > 0:
                    vis_vox.append(cam_lookup_table[ix][iy])
        return vis_vox

    def run_voxel_reconstruction(self, masks):
        # masks shape: (4, 428, 486, 644)
        # for every camera, for every frame in camera, a mask with white pixels in foreground and black pixels in background
        # Lookup table will contain for every camera, for every pixel value, a list of voxel coords which are projected to that pixel value

        lookup_table = self.create_lookup_table()

        for frame in range(len(masks[0])):
            cam_vis_vox = [self.return_visible_voxels(masks[i][frame], lookup_table[i]) for i in range(4)]
            all_vis_vox = np.logical_and(np.logical_and(np.logical_and(cam_vis_vox[0], cam_vis_vox[1]), cam_vis_vox[2]),
                                         cam_vis_vox[3])
            # TODO visualize all_vis_vox for this frame
