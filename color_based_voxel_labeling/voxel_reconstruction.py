import numpy as np
import cv2 as cv
import pickle
import executable as Executable
import assignment as Assignment
#import surface_mesh as Mesh
from engine.config import config
import copy


class VoxelReconstruction:
    rotation_vectors = []
    translation_vectors = []
    intrinsics = []
    dist_mtx = []
    lookup_table = []
    stepsize = 0


    def __init__(self, params) -> None:
        self.rotation_vectors = params['rotation_vectors']
        self.translation_vectors = params['translation_vectors']
        self.intrinsics = params['intrinsics']
        self.dist_mtx = params['dist_mtx']
        self.stepsize = params['stepsize']
        self.remove_ghosts = params['ghosts']

        Assignment.load_parameters_from_pickle(params['path'])
        self.initialise_all_voxels()

    def initialise_all_voxels(self):
        self.cam_coords = np.array(Assignment.get_cam_positions())
        [max_x, max_y, max_z] = np.max(self.cam_coords, axis=0).astype(int)
        self.xb = max_x // self.stepsize
        self.zb = max_z // self.stepsize
        self.yb = max_y // self.stepsize
        self.all_voxels = self.stepsize * np.array(
            [[x, y, z] for x in range(-self.xb, self.xb) for y in range(0, 2 * self.yb) for z in
             range(-self.zb, self.zb)])
        self.cams_vis_vox_indices = np.tile(np.zeros(len(self.all_voxels)), (4, 1))
        self.cams_pos_vis_vox_indices = copy.deepcopy(self.cams_vis_vox_indices)


    def compute_xyz_index(self, vox):
        # For a given voxel in all_voxels, compute the corresponding index of that voxel in all_voxels
        [x, y, z] = vox
        return 4 * self.yb * self.zb * (
                x // self.stepsize + self.xb) + 2 * self.zb * y // self.stepsize + (
                z // self.stepsize + self.zb)

    def create_lookup_table(self):
        # The lookup_table shape is (4,644,486) for indexing the cameras and the pixels of each camera.
        # Each pixels stores a variable sized list of multiple [x,y,z] coordinates.
        lookup_table = [[[[] for _ in range(486)] for _ in range(644)] for _ in range(4)]
        for cam in range(4):
            print('cam: ', cam)
            self.all_voxels = self.all_voxels
            float_all_voxels = np.float64([
                    self.all_voxels[:, 0], 
                    self.all_voxels[:, 2],
                    -self.all_voxels[:, 1]
            ])
            
            idx = cv.projectPoints(
                float_all_voxels, 
                self.rotation_vectors[cam],
                self.translation_vectors[cam],
                self.intrinsics[cam], 
                distCoeffs=self.dist_mtx[cam]
            )
            ix = idx[0][:, 0][:, 0]
            iy = idx[0][:, 0][:, 1]

            iiy = np.asarray(abs(iy) < 486).nonzero()
            iix = np.asarray(abs(ix) < 644).nonzero()

            indices = np.intersect1d(iix, iiy)

            ix = np.take(ix, indices).astype(int)
            iy = np.take(iy, indices).astype(int)

            voxels = np.take(self.all_voxels, indices, 0)

            for index in range(len(voxels)):
                lookup_table[cam][ix[index]][iy[index]].append(voxels[index])

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

    def pixels_to_xyz_indices(self, pixels, cam):
        xyz_indices = []
        for i in range(len(pixels[0])):
            iy = pixels[0][i]
            ix = pixels[1][i]
            for vox in self.lookup_table[cam][ix][iy]:
                xyz_indices.append(self.compute_xyz_index(vox))
        return np.array(xyz_indices, dtype = int)


    def cam_dist(self, cam, vox):
        return np.linalg.norm(self.cam_coords[cam]-vox)

    def create_distance_table(self):
        #method to compute the distance of each voxel to each camera
        self.distance_table = np.array([[self.cam_dist(i,vox) for vox in self.all_voxels] for i in range(4)])



    def possible_visible_voxels_per_cam(self, masks, num_cameras):
        for cam in range(num_cameras):
            for iy in range(len(masks[cam])):
                for ix in range(len(masks[cam])):
                    for vox in self.lookup_table[cam][ix][iy]:
                        xyz_index = self.compute_xyz_index(vox)
                        self.cams_pos_vis_vox_indices[cam][xyz_index] = 1

        return self.cams_pos_vis_vox_indices

    def index_visible_voxels(self):
        return np.ravel(
            np.argwhere(
                (self.cams_pos_vis_vox_indices >= self.cams_vis_vox_indices)
                    .all(axis = 0)
            )
        )

    def reconstruct_voxels(self, masks, prev_masks, frame_num):
        # masks shape: (4, 486, 644). Reconstructs a single frame
        print('frame ' + str(frame_num))
        num_cameras = 4
        self.possible_visible_voxels_per_cam(masks, num_cameras)

        if frame_num == 0:
            for cam in range(num_cameras):
                cam_indices = np.nonzero(masks[cam])
                for i in range(len(cam_indices[0])):
                    iy = cam_indices[0][i]
                    ix = cam_indices[1][i]
                    for vox in self.lookup_table[cam][ix][iy]:
                        xyz_index = self.compute_xyz_index(vox)
                        self.cams_vis_vox_indices[cam][xyz_index] = 1
        else:
            for cam in range(num_cameras):
                xor = np.logical_xor(prev_masks[cam], masks[cam])
                removed_pixels = np.nonzero(np.logical_and(xor, prev_masks[cam]))
                added_pixels = np.nonzero(np.logical_and(xor, masks[cam]))

                removed_xyz_indices = self.pixels_to_xyz_indices(removed_pixels, cam)
                self.cams_vis_vox_indices[cam][removed_xyz_indices] = 0

                added_xyz_indices = self.pixels_to_xyz_indices(added_pixels, cam)
                self.cams_vis_vox_indices[cam][added_xyz_indices] = 1

        # Eerst weer duidelijk krijgen wat nou precies de bodoeling is voor deze
        # if self.remove_ghosts:
        #     k= 100 #number of voxels to keep per camera
        #     visible_distances = self.distance_table * self.cams_vis_vox_indices
        #     visible_distances[visible_distances==0] =np.Inf #the zero distances are not visible. want them to be at the back of the sorted array
        #     sorted_voxel_indices_on_cam_distance = np.argsort(visible_distances)
        #     k_closest_indices = sorted_voxel_indices_on_cam_distance[:,:k]

        self.all_vis_voxels = self.all_voxels[self.index_visible_voxels()]
        return self.all_vis_voxels

