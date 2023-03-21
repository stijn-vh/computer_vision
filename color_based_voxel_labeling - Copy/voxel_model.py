import numpy as np
import cv2 as cv

class VoxelModel:
    #Base class for both VoxelReconstruction and ColorModel
    def __init__(self, params) -> None:
        self.rotation_vectors = params['rotation_vectors']
        self.translation_vectors = params['translation_vectors']
        self.intrinsics = params['intrinsics']
        self.dist_mtx = params['dist_mtx']
        self.stepsize = params['stepsize']
        self.xb = params['xb']
        self.yb = params['yb']
        self.zb = params['zb']

    def compute_xyz_index(self, vox):
        # For a given voxel in all_voxels, compute the corresponding index of that voxel in all_voxels
        [x, y, z] = vox
        return 4 * self.yb * self.zb * (
                x // self.stepsize + self.xb) + 2 * self.zb * y // self.stepsize + (
                z // self.stepsize + self.zb)

    def get_pixels_in_range(self, ix,iy):
        iix = np.asarray(abs(ix) < 644).nonzero()
        iiy = np.asarray(abs(iy) < 486).nonzero()
        indices = np.intersect1d(iix, iiy)
        ix = np.take(ix, indices).astype(int)
        iy = np.take(iy, indices).astype(int)
        return ix,iy,indices

    def project_voxels(self, voxels, cam):
        #Returns two lists, one containing the x coords and one the y coords of the voxels projected to the specific cam
        transformed_voxels = np.float64([
            voxels[:, 0],
            voxels[:, 2],
            -voxels[:, 1]
        ])
        idx = cv.projectPoints(
            transformed_voxels,
            self.rotation_vectors[cam],
            self.translation_vectors[cam],
            self.intrinsics[cam],
            distCoeffs=self.dist_mtx[cam]
        )
        ix = idx[0][:, 0][:, 0].astype(int)
        iy = idx[0][:, 0][:, 1].astype(int)
        return ix,iy
