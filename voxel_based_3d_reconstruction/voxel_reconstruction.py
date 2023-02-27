import numpy as np
import cv2 as cv
import pickle
import reconstruction.assignment as Assignment
import reconstruction.executable as Executable

class VoxelReconstruction:
    # TODO method which reads in these values
    rotation_vectors = []
    translation_vectors = []
    intrinsics = []

    def __init__(self, path) -> None:
        with open(path, 'rb') as f:
            camera_params = pickle.load(f)

            for camera in camera_params:
                self.rotation_vectors.append(camera_params[camera]['extrinsic_rvec'])
                self.translation_vectors.append(camera_params[camera]['extrinsic_tvec'])
                self.intrinsics.append(camera_params[camera]['intrinsic_mtx'])
        
        Assignment.load_parameters_from_pickle(path)

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
        vis_vox = []
        nonzeros = np.nonzero(mask)
        for ix in nonzeros[0]:
            for iy in nonzeros[1]:
                for vox in cam_lookup_table[ix][iy]:
                    vis_vox.append(vox)
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
<<<<<<< Updated upstream
            
            Assignment.voxels[frame] = all_vis_vox
            Executable.main()
    
=======
            # TODO visualize all_vis_vox for this frame
            
>>>>>>> Stashed changes
