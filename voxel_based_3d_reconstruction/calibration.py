import os
import random
import cv2 as cv
import numpy as np
import helpers.offline_phase as OfflinePhase
from bs4 import BeautifulSoup

class Calibration:
    config = {
        'width': 0,
        'height': 0,
        'size': 0,
        'amount_of_frames_to_read': 5
    }

    cameras = {
        'cam1': {}, 
        'cam2': {}, 
        'cam3': {}, 
        'cam4': {}
    }

    def __init__(self) -> None:
        self.load_config()
        self.set_offline_phase_config()

    def set_offline_phase_config(self):
        objp = np.zeros((self.config['width'] * self.config['height'], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.config['width'], 0:self.config['height']].T.reshape(-1, 2)

        OfflinePhase.set_config({
            'criteria': (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001),
            'image_name': 'current_frame',
            'num_cols': self.config['width'],
            'num_rows': self.config['height'],
            'objp': objp
        })

    def handle_frame_from_video(self, video, totalFrames):
        while(True):
            randomFrameNumber = random.randint(0, totalFrames)
            video.set(cv.CAP_PROP_POS_FRAMES, randomFrameNumber)
            s, frame = video.read()

            if s:
                succeeded = OfflinePhase.handle_image(frame, canDeterminePointsManually = False)

                if succeeded:
                    break

    def obtain_intrinsics_from_cameras(self):
        for cam_name in self.cameras:
            video = cv.VideoCapture(os.path.dirname(__file__) + '\data\data\\' + cam_name + '\intrinsics.avi')

            totalFrames = video.get(cv.CAP_PROP_FRAME_COUNT)

            for i in range(self.config['amount_of_frames_to_read']):
                self.handle_frame_from_video(video, totalFrames)

            w = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
            h = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))

            ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
                OfflinePhase.objpoints, 
                OfflinePhase.imgpoints, 
                [w, h][::-1], None, None)

            self.cameras[cam_name]['img_points'] = OfflinePhase.imgpoints
            self.cameras[cam_name]['obj_points'] = OfflinePhase.objpoints
            self.cameras[cam_name]['intrinsic_mtx'] = mtx
            self.cameras[cam_name]['intrinsic_dist'] = dist

            OfflinePhase.imgpoints = []
            OfflinePhase.objpoints = []

        return self.cameras

    def obtain_extrinsics_from_cameras(self):
        for cam_name in self.cameras:

            print(np.array(self.cameras[cam_name]['obj_points']).shape)
            print(np.array(self.cameras[cam_name]['img_points']).shape)
            
            # TODO:
            # First use findChessBoardCorners on frame of checkerboard.avi, 
            # maybe optimize frame beforehand
            # use same frame for all cams?
            # use imgPoints from this call togheter with int_mtx/dist to solvePnP
            rvec, tvec = cv.solvePnP(
                self.cameras[cam_name]['obj_points'], 
                self.cameras[cam_name]['img_points'], 
                self.cameras[cam_name]['intrinsic_mtx'], 
                self.cameras[cam_name]['intrinsic_dist'], 
                useExtrinsicGuess = False
            )

            self.cameras[cam_name]['extrinsic_rvec'] = rvec
            self.cameras[cam_name]['extrinsic_tvec'] = tvec
        
        return




    def write_to_config(self, text):
        return

    def load_config(self):
        file_path = os.path.dirname(__file__) + '\data\data\checkerboard.xml'

        with open(file_path, 'r') as f:
            c = f.read()

            bs_data = BeautifulSoup(c, 'xml')

            self.config['width'] = int(bs_data.find('CheckerBoardWidth').contents[0])
            self.config['height'] = int(bs_data.find('CheckerBoardHeight').contents[0])
            self.config['size'] = int(bs_data.find('CheckerBoardSquareSize').contents[0])