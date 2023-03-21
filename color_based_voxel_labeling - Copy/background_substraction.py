import os
import cv2 as cv
import numpy as np
import sklearn as sk


class BackgroundSubstraction:
    cam_means = []
    thresholds = []
    cam_std_devs = []
    num_contours = []

    def __init__(self) -> None:
        self.thresholds = np.array([
            [10, 2, 18],
            [10, 2, 14],
            [10, 1, 14],
            [10, 2, 20]]
        )
        
        self.num_contours = [4, 5, 5, 6]

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

        for f in folders:
            means, std_devs = self.compute_background_values('\data\\' + f + '\\background.avi')
            self.cam_means.append(means)
            self.cam_std_devs.append(std_devs)

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

    # Helper function for gridsearch
    def evaluate_mask(self, mask, i):
        if i == 0:
            thresh = 247
        else:
            thresh = 255
        path = os.path.dirname(__file__) + "\\cam" + str(i + 1) + "_groundtruth.png"
        groundtruth = cv.imread(path)
        groundtruth = (255 * (cv.cvtColor(groundtruth, cv.COLOR_BGR2GRAY) >= thresh)).astype(np.uint8)
        xor = np.logical_xor(mask, groundtruth)
        in_gt_not_in_mask = np.logical_and(xor, groundtruth)
        in_mask_not_in_gt = np.logical_and(xor, mask)
        return 1.5 * np.sum(in_gt_not_in_mask) + np.sum(in_mask_not_in_gt)  # The lower the score the better

    def gridsearch(self, cam_means, cam_std_devs):
        cam_folders = ['cam1', 'cam2', 'cam3', 'cam4']
        thresholds = []
        for i_ in np.arange(1, 12):
            for j_ in np.arange(1, 8):
                for k_ in 2 * np.arange(3, 15):
                    thresholds.append([i_, j_, k_])
        num_contours = np.arange(1, 3)
        best_num_contours = np.repeat(-1, 4)
        best_thresholds = np.tile([-1, -1, -1], (4, 1))
        for i, f in enumerate(cam_folders):
            best_score = float('inf')
            for j in range(len(num_contours)):
                for k in range(len(thresholds)):
                    print("trying thresh", thresholds[k])
                    mask = self.compute_mask(thresholds[k], num_contours[j], '\data\\' + f + '\\video.avi',
                                             cam_means[i],
                                             cam_std_devs[i])
                    score = self.evaluate_mask(mask, i)
                    if score < best_score:
                        best_score = score
                        best_num_contours[i] = num_contours[j]
                        best_thresholds[i] = thresholds[k]
        return best_thresholds, best_num_contours

    def compute_mask_in_frame(self, frame, cam_number):
        mask = np.float32(255 * (np.sum(np.abs(
            frame - self.cam_means[cam_number]) > 
            self.thresholds[cam_number] * (self.cam_std_devs[cam_number] + 0.1), 2) == 3)
        )

        mask = mask.astype(np.uint8)

        contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        (height, width) = mask.shape
        mask = np.zeros((height, width), np.uint8)
        contours = sorted(contours, key=cv.contourArea, reverse=True)

        cv.drawContours(mask,
                        contours[:self.num_contours[cam_number]], -1, 255,
                        thickness=cv.FILLED)

        return mask
    
    # Used in the actual background substraction
    def compute_masks(self, thresholds, num_contour, pathname, cam_mean, cam_std_dev, show_video):
        video = cv.VideoCapture(os.path.dirname(__file__) + pathname)
        masks = []
        frames = []
        ret = True
        #needs to be made offline to handle frame by frame
        max_num = 100
        num = 0
        while ret & (num < max_num):
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
                frames.append(frame)
                num += 1
                if show_video:
                    cv.imshow("video", mask)
                    cv.waitKey(1)
        return masks, frames

    def background_subtraction(self, thresholds, num_contours, cam_means, cam_std_devs, show_video):

        self.compute_mask_in_frame()
        folders = ['cam1', 'cam2', 'cam3', 'cam4']
        camera_masks = []
        camera_frames = []
        for i, f in enumerate(folders):
            masks, frames = self.compute_masks(thresholds[i], num_contours[i], '\data\\' + f + '\\video.avi',
                                               cam_means[i],
                                               cam_std_devs[i], show_video)
            camera_masks.append(masks)
            camera_frames.append(frames)
        return camera_masks, camera_frames
