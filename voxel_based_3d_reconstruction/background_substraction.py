import os
import cv2 as cv
import numpy as np
import sklearn as sk


class BackgroundSubstraction:
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
            means, std_devs = self.compute_background_values('\data\\' + f + '\\background.avi')
            cam_means.append(means)
            cam_std_devs.append(std_devs)
        return cam_means, cam_std_devs

    # helper func for gridsearch
    def compute_mask(self, thresholds, num_contour, pathname, cam_mean, cam_std_dev):
        video = cv.VideoCapture(os.path.dirname(__file__) + pathname)
        ret, frame = self.read_video(video)
        if ret:
            mask = np.float32(255 * (np.sum(np.abs(frame - cam_mean) > thresholds * (cam_std_dev + 0.1), 2) == 3))
            mask = mask.astype(np.uint8)
            contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            (height, width) = mask.shape
            mask = np.zeros((height, width), np.uint8)
            contours = sorted(contours, key=cv.contourArea, reverse=True)
            cv.drawContours(mask,
                            contours[:num_contour], -1, 255,
                            thickness=cv.FILLED)
        return mask
    #Helper function for gridsearch
    def evaluate_mask(self, mask, i):
        if i == 0:
            thresh = 247
        else:
            thresh = 255
        path = os.path.dirname(__file__) + "\\cam" + str(i + 1) + "_groundtruth.png"
        groundtruth = cv.imread(path)
        groundtruth = (255 * (cv.cvtColor(groundtruth, cv.COLOR_BGR2GRAY) >= thresh)).astype(np.uint8)
        return np.sum(np.logical_xor(mask, groundtruth))  # The lower the score the better

    def gridsearch(self, cam_means, cam_std_devs):
        folders = ['cam1', 'cam2', 'cam3', 'cam4']
        thresholds = []
        for i_ in np.arange(1, 15):
            for j_ in np.arange(1, 15):
                for k_ in 2*np.arange(1, 15):
                    thresholds.append([i_, j_, k_])
        num_contours = np.arange(1, 5)
        best_num_contours = np.repeat(-1, 4)
        best_thresholds = np.tile([-1, -1, -1], (4, 1))
        for i, f in enumerate(folders):
            best_score = float('inf')
            for j in range(len(num_contours)):
                for k in range(len(thresholds)):
                    mask = self.compute_mask(thresholds[k], num_contours[j], '\data\\' + f + '\\video.avi',
                                             cam_means[i],
                                             cam_std_devs[i])
                    score = self.evaluate_mask(mask, i)
                    if score < best_score:
                        best_score = score
                        best_num_contours[i] = num_contours[j]
                        best_thresholds[i] = thresholds[k]
        print(best_thresholds)
        print(best_num_contours)

    #Used in the actual background substraction
    def compute_masks(self, thresholds, num_contour, pathname, cam_mean, cam_std_dev, show_video):
        video = cv.VideoCapture(os.path.dirname(__file__) + pathname)
        masks = []
        ret = True
        while ret:
            ret, frame = self.read_video(video)
            if ret:
                mask = np.float32(255 * (np.sum(np.abs(frame - cam_mean) > thresholds * (cam_std_dev + 0.1), 2) == 3))
                mask = mask.astype(np.uint8)
                contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
                (height, width) = mask.shape
                mask = np.zeros((height, width), np.uint8)
                contours = sorted(contours, key=cv.contourArea, reverse=True)
                cv.drawContours(mask,
                                contours[:num_contour], -1, 255,
                                thickness=cv.FILLED)
                masks.append(mask)
                if show_video:
                    cv.imshow("video", mask)
                    cv.waitKey(1)
        return masks

    def background_subtraction(self, thresholds, num_contours, cam_means, cam_std_devs, show_video):
        folders = ['cam1', 'cam2', 'cam3', 'cam4']
        camera_masks = []
        for i, f in enumerate(folders):
            masks = self.compute_masks(thresholds[i], num_contours[i], '\data\\' + f + '\\video.avi', cam_means[i],
                                       cam_std_devs[i], show_video)
            camera_masks.append(masks)
        return camera_masks
