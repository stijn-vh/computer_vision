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

    def __init__(self) -> None:
        self.load_config()
        self.set_offline_phase_config()

        pass

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

    def handle_frame_from_video(self, video, totalFrames, i):
        randomFrameNumber = random.randint(0, totalFrames)
        video.set(cv.CAP_PROP_POS_FRAMES, randomFrameNumber)
        s, frame = video.read()

        if s:
            succ = OfflinePhase.handle_image(frame, canDeterminePointsManually = False)

            if succ == False:
                i = i - 1
        
        return i

    def load_frames_from_videos(self):
        folders = ['cam1', 'cam2', 'cam3', 'cam4']

        for f in folders:
            video = cv.VideoCapture(os.path.dirname(__file__) + '\data\data\\' + f + '\intrinsics.avi')

            totalFrames = video.get(cv.CAP_PROP_FRAME_COUNT)

            for i in range(self.config['amount_of_frames_to_read']):
                i = self.handle_frame_from_video(video, totalFrames, i)
        
        print(OfflinePhase.objpoints)
        print(OfflinePhase.imgpoints)

        return 

    def obtain_intrinsics_from_camera(self):
        return

    def calculate_extrinsics(self):
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