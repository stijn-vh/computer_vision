import numpy as np
import cv2 as cv
import pickle
import executable as Executable
import assignment as Assignment
from engine.config import config


class VoxelReconstruction:
    rotation_vectors = []
    translation_vectors = []
    intrinsics = []
    dist_mtx = []
    lookup_table = []
    # change to 3D array
    # vis_vox_indices = np.ones(config['world_width'] * config['world_depth'] * config['world_width'])
    prev_masks = None

    def __init__(self, path) -> None:

        with open(path, 'rb') as f:
            camera_params = pickle.load(f)

            for camera in camera_params:
                self.rotation_vectors.append(np.array(
                    [camera_params[camera]['extrinsic_rvec'][0], camera_params[camera]['extrinsic_rvec'][1],
                     camera_params[camera]['extrinsic_rvec'][2]]))
                self.translation_vectors.append(np.array(
                    [camera_params[camera]['extrinsic_tvec'][0], camera_params[camera]['extrinsic_tvec'][1],
                     camera_params[camera]['extrinsic_tvec'][2]]))
                self.intrinsics.append(camera_params[camera]['intrinsic_mtx'])
                self.dist_mtx.append(camera_params[camera]['intrinsic_dist'])

        Assignment.load_parameters_from_pickle(path)
        cam_coords = Assignment.get_cam_positions()
        [max_x, max_y, max_z] = np.max(cam_coords, axis=0).astype(int)
        [min_x, min_y, min_z] = np.min(cam_coords, axis =0).astype(int)
        stepsize = 1
        self.xb = max_x//2
        self.zb = max_z//2
        self.yb = max_y
        self.all_voxels = np.array([[x, y, z] for x in range(-self.xb, self.xb, stepsize) for y in range(0, 2*self.yb, stepsize) for z in range(-self.zb, self.zb, stepsize)])
        self.vis_vox_indices = np.ones(self.all_voxels.shape[0]) #Will use ANDs so initiliased as ones

    def create_lookup_table(self):
        # The lookup_table shape is (4,644,486) for indexing the cameras and the pixels of each camera.
        # Each pixels stores a variable sized list of multiple [x,y,z] coordinates.
        lookup_table = [[[[] for _ in range(486)] for _ in range(644)] for _ in range(4)]
        for cam in range(4):
            for [x,y,z] in self.all_voxels:

                idx = cv.projectPoints(np.float64([x,z,-y]), self.rotation_vectors[cam],
                                       self.translation_vectors[cam],
                                       self.intrinsics[cam], distCoeffs=self.dist_mtx[cam])
                ix = idx[0][0][0][0].astype(int)
                iy = idx[0][0][0][1].astype(int)
                if -644 < ix < 644 and -486 < iy < 486:
                    lookup_table[cam][ix][iy].append([x,y,z])
        return lookup_table

    def return_visible_voxels(self, mask, cam_lookup_table):
        # mask has shape (486,644) corresponding to the pixels in a single frame
        vis_vox = []
        nonzeros = np.nonzero(mask)
        for ix in nonzeros[0]:
            for iy in nonzeros[1]:
                for vox in cam_lookup_table[ix][iy]:
                    vis_vox.append(vox)
        return vis_vox

    def test_voxel_reconstruction(self, masks):
        frame = 0
        num_cams = 4
        for [x, y, z] in self.all_voxels:
            num_seen = 0
            for cam in range(num_cams):
                cam_img_idx = cv.projectPoints(np.float64([x, z, -y]), self.rotation_vectors[cam],
                                               self.translation_vectors[cam],
                                               self.intrinsics[cam],
                                               distCoeffs=self.dist_mtx[cam])  # np.array([])  # distCoeffs=self.dist_mtx[cam] #np.array([])
                ix = cam_img_idx[0][0][0][0].astype(int)
                iy = cam_img_idx[0][0][0][1].astype(int)
                if -644 < ix < 644 and -486 < iy < 486:
                    if masks[cam][frame][iy][ix] == 255:
                        num_seen += 1
            if num_seen == num_cams:
                self.all_vis_voxels.append([x, y, z])
        print("all vis voxels:", self.all_vis_voxels)
        Assignment.voxels = self.all_vis_voxels
        Executable.main()

    def run_voxel_reconstruction(self, masks):
        # masks shape: (4, 428, 486, 644)
        # for every camera, for every frame in camera, a mask with white pixels in foreground and black pixels in background
        # 486 y pixels and 644 x pixels
        # Lookup table will contain for every camera, for every pixel value, a list of voxel coords which are projected to that pixel value

        num_frames = 1
        num_cameras = 4

        # for frame in range(len(masks[0])):
        for frame in range(num_frames):
            # For the first run
            if frame == 0:
                for cam in range(num_cameras):
                    cam_vis_vox_indices = np.zeros(len(self.vis_vox_indices))
                    cam_indices = np.nonzero(masks[cam][frame])
                    for i in range(len(cam_indices[0])):
                        iy = cam_indices[0][i]
                        ix = cam_indices[1][i]
                        for [x, y, z] in self.lookup_table[cam][ix][iy]:
                            xyz_index = 4*self.yb*self.zb*(x+self.xb) + 2*self.zb*y + (z + self.zb)
                            cam_vis_vox_indices[xyz_index] = True
                    self.vis_vox_indices = np.logical_and(self.vis_vox_indices, cam_vis_vox_indices)
                self.all_vis_voxels = self.all_voxels[self.vis_vox_indices]
            else:
                for cam in range(num_cameras):
                    new_vis_vox_indices = np.zeros(len(self.vis_vox_indices))
                    xor = np.logical_xor(masks[cam][frame - 1], masks[cam][frame])
                    removed_pixels = np.nonzero(np.logical_and(xor, masks[cam][frame - 1]))
                    added_pixels = np.nonzero(np.logical_and(xor, masks[cam][frame]))
                    for i in range(len(removed_pixels[0])):
                        iy = removed_pixels[0][i]
                        ix = removed_pixels[1][i]
                        for [x, y, z] in self.lookup_table[cam][ix][iy]:
                            xyz_index = 4 * self.yb * self.zb * (x + self.xb) + 2 * self.zb * y + (z + self.zb)
                            self.vis_vox_indices[xyz_index] = False
                    for i in range(len(added_pixels[0])):
                        iy = added_pixels[0][i]
                        ix = added_pixels[1][i]
                        for [x, y, z] in self.lookup_table[cam][ix][iy]:
                            xyz_index = 4 * self.yb * self.zb * (x + self.xb) + 2 * self.zb * y + (z + self.zb)
                            new_vis_vox_indices[xyz_index] += 1
                self.all_vis_voxels = np.append(self.all_vis_voxels, self.all_voxels[new_vis_vox_indices == 4], 0)
            Assignment.voxels = self.all_vis_voxels
            Executable.main()
