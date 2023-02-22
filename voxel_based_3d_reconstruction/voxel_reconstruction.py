import os
import cv2 as cv
import numpy as np
import sklearn as sk


class VoxelReconstruction:
    def __init__(self) -> None:
        pass

    def read_video(self, video):
        ret, frame = video.read()
        if ret:
            frame = np.float32(cv.cvtColor(frame, cv.COLOR_BGR2HSV))
        return ret, frame

    def compute_background_values(self, pathname):
        video = cv.VideoCapture(os.path.dirname(__file__) + pathname)
        ret, frame = self.read_video(video)
        sums = frame
        squared_sums = frame ** 2
        count = 1
        while ret:
            ret, frame = self.read_video(video)
            if ret:
                sums += frame
                squared_sums += frame ** 2
                count += 1
        means = sums / count
        std_devs = np.sqrt(squared_sums / count - means ** 2)
        return means, std_devs

    def create_background_model(self):
        folders = ['cam1', 'cam2', 'cam3', 'cam4']
        cam_means = []
        cam_std_devs = []
        for f in folders:
            means, std_devs = self.compute_background_values('\data\data\\' + f + '\\background.avi')
            cam_means.append(means)
            cam_std_devs.append(std_devs)
        return cam_means, cam_std_devs

    def compute_masks(self, thresholds, pathname, cam_mean, cam_std_dev):
        video = cv.VideoCapture(os.path.dirname(__file__) + pathname)
        masks = []
        ret = True
        while ret:
            ret, frame = self.read_video(video)
            if ret:
                mask = np.float32(255 * (np.sum(np.abs(frame - cam_mean) > thresholds * (cam_std_dev+0.1), 2) == 3))
                #kernel = np.ones((1, 1), np.uint8)
                #mask = cv.erode(mask, kernel, iterations=1)
                masks.append(mask)
                #cv.imshow("video", mask)
                #cv.waitKey(1)
        return masks

    def background_subtraction(self, cam_means, cam_std_devs):
        thresholds = [10, 20, 20]
        folders = ['cam1', 'cam2', 'cam3', 'cam4']
        for i, f in enumerate(folders):
            masks = self. compute_masks(thresholds, '\data\data\\' + f + '\\video.avi', cam_means[i], cam_std_devs[i])

        return
