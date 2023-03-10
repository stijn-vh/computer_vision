import os
import random
import pickle
import cv2 as cv
import numpy as np
import helpers.offline_phase as OfflinePhase
import helpers.online_phase as OnlinePhase
from bs4 import BeautifulSoup


class Calibration:
    config = {
        'width': 0,
        'height': 0,
        'size': 0,
        'amount_of_frames_to_read': 25
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
        #Chessboard coordinates are x and z
        objp = np.zeros((self.config['width'] * self.config['height'], 3), np.float32)
        objp[:,:2] = self.config['size']/10 * np.mgrid[0:self.config['width'], 0:self.config['height']].T.reshape(-1, 2)

        c = {
            'criteria': (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001),
            'image_name': 'current_frame',
            'num_cols': self.config['width'],
            'num_rows': self.config['height'],
            'objp': objp
        }

        OfflinePhase.set_config(c)
        OnlinePhase.set_config(c)

    def handle_frame_from_video(self, video, totalFrames, cam_name=None):
        r_numbers = []
        while (True):
            randomFrameNumber = random.randint(0, totalFrames)
            video.set(cv.CAP_PROP_POS_FRAMES, randomFrameNumber)
            s, frame = video.read()
            if s and randomFrameNumber not in r_numbers:
                r_numbers.append(randomFrameNumber)
                succeeded = OfflinePhase.handle_image(frame, 1, canDeterminePointsManually=False)

                if succeeded:
                    break

    def loop_through_video_frames(self, video, cam_name):
        f_count = 0

        while (True):
            s, frame = video.read()

            if s:
                succeeded = OfflinePhase.handle_image(frame, 200, canDeterminePointsManually=False)

                if succeeded:
                    print(cam_name, ', frame: ', f_count)
                    return frame

            if f_count == video.get(cv.CAP_PROP_FRAME_COUNT):
                print('no auto-frame found for ', cam_name)
                randomFrameNumber = random.randint(0, video.get(cv.CAP_PROP_FRAME_COUNT))
                video.set(cv.CAP_PROP_POS_FRAMES, randomFrameNumber)
                s, frame = video.read()
                succeeded = OfflinePhase.handle_image(frame, 200, canDeterminePointsManually=True)
                return frame

            f_count = f_count + 1

    def calibrate_camera(self, video, cam_name):
        w = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
        h = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))

        if cam_name is 'cam4':
            for i in range(len(OfflinePhase.imgpoints)):
                OfflinePhase.imgpoints[i] = OfflinePhase.imgpoints[i][::-1]

        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
            OfflinePhase.objpoints,
            OfflinePhase.imgpoints,
            [w, h][::-1], None, None)

        self.cameras[cam_name]['intrinsic_mtx'] = mtx
        self.cameras[cam_name]['intrinsic_dist'] = dist

    def obtain_intrinsics_from_cameras(self):
        for cam_name in self.cameras:
            if cam_name != 'cam1':
                continue

            OfflinePhase.cam_name = cam_name

            video = cv.VideoCapture(os.path.dirname(__file__) + '\data\\' + cam_name + '\intrinsics.avi')

            totalFrames = video.get(cv.CAP_PROP_FRAME_COUNT)

            for i in range(self.config['amount_of_frames_to_read']):
                self.handle_frame_from_video(video, totalFrames)

            self.calibrate_camera(video, cam_name)

            OfflinePhase.imgpoints = []
            OfflinePhase.objpoints = []

        return self.cameras

    def calculate_extrinsics(self, cam_name):
        objp = OfflinePhase.objpoints[0]

        r, rvec, tvec = cv.solvePnP(
            objp,
            OfflinePhase.imgpoints[0],
            self.cameras[cam_name]['intrinsic_mtx'],
            self.cameras[cam_name]['intrinsic_dist'],
            useExtrinsicGuess=False
        )

        self.cameras[cam_name]['extrinsic_rvec'] = rvec
        self.cameras[cam_name]['extrinsic_tvec'] = tvec
        self.cameras[cam_name]['R'] = cv.Rodrigues(rvec)

    def obtain_extrinsics_from_cameras(self):
        for cam_name in self.cameras:
            if cam_name != 'cam1':
                continue
            
            video = cv.VideoCapture(os.path.dirname(__file__) + '\data\\' + cam_name + '\checkerboard.avi')

            s, frame = video.read()


            if s:
                OfflinePhase.handle_image(frame, time=5000, shouldDetermineManually = True)

            self.calculate_extrinsics(cam_name)

            OnlinePhase.name = cam_name

            OnlinePhase.draw_on_image({
                'mtx': self.cameras[cam_name]['intrinsic_mtx'],
                'dist': self.cameras[cam_name]['intrinsic_dist']
            }, frame, OfflinePhase.imgpoints[0])

            OfflinePhase.imgpoints = []
            OfflinePhase.objpoints = []

    def write_to_config(self, text):
        return

    def load_config(self):
        file_path = os.path.dirname(__file__) + '\data\checkerboard.xml'

        with open(file_path, 'r') as f:
            c = f.read()

            bs_data = BeautifulSoup(c, 'xml')

            self.config['width'] = int(bs_data.find('CheckerBoardWidth').contents[0])
            self.config['height'] = int(bs_data.find('CheckerBoardHeight').contents[0])
            self.config['size'] = int(bs_data.find('CheckerBoardSquareSize').contents[0])
