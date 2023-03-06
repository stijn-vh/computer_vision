import numpy as np
import cv2 as cv
import os


class AutoBackgroundSubstraction:
    backSubs = [cv.createBackgroundSubtractorMOG2() for i in range(4)]

    def __init__(self) -> None:
        pass

    def read_video(self, video):
        ret, frame = video.read()
        if ret:
            frame = np.float32(cv.cvtColor(frame, cv.COLOR_BGR2HSV))
        return ret, frame

    def create_background_models(self, pathnames):
        for cam in range(4):
            video = cv.VideoCapture(os.path.dirname(__file__) + pathnames[cam])
            ret, frame = self.read_video(video)
            self.backSubs[cam].apply(frame)
            while ret:
                ret, frame = self.read_video(video)
                if ret:
                    self.backSubs[cam].apply(frame)

    def compute_masks(self, pathnames, show_video):
        masks = np.zeros((4, 428, 486, 644))
        for cam in range(4):
            video = cv.VideoCapture(os.path.dirname(__file__) + pathnames[cam])
            frame_num = 0
            ret, frame = self.read_video(video)
            mask = self.backSubs[cam].apply(frame, 0)
            masks[cam][frame_num] = mask
            frame_num += 1
            while ret:
                ret, frame = self.read_video(video)
                if ret:
                    self.backSubs[cam].apply(frame,0)
                    masks[cam][frame_num] = mask
                    frame_num += 1
                    if show_video:
                        cv.imshow("video", mask)
                        cv.waitKey(1)
        return masks

    def background_subtraction(self, show_video):
        videopaths = ['\data\\cam' + i + '\\video.avi' for i in ['1', '2', '3', '4']]
        backgroundpaths = ['\data\\cam' + i + '\\background.avi' for i in ['1', '2', '3', '4']]
        self.create_background_models(backgroundpaths)
        masks = self.compute_masks(videopaths, show_video)
        return masks
