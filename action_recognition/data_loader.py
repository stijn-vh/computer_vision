"""## Read the train and test splits, combine them and make better splits to help training networks easier."""

from collections import Counter
from sklearn.model_selection import train_test_split
import os
import glob
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import cv2 as cv
import numpy as np
import re


class DataLoader:
    def __init__(self, data_params):
        self.image_size = data_params["image_size"]
        self.batch_size = data_params["batch_size"]
        self.print_info = data_params["print_info"]
        self.test_size = data_params["test_size"]
        self.val_size = data_params["val_size"]
        self.names_to_ints = None
        self.data_directory = None
        self.train_augmentation = {"rotation_range": 10,
                                   "horizontal_flip": True,
                                   "width_shift_range": 0.2,
                                   "height_shift_range": 0.2,
                                   "zoom_range": 0.2,
                                   "brightness_range": [0.7, 1.3]}
        self.no_augmentation = {"rotation_range": 0,
                                "horizontal_flip": False,
                                "width_shift_range": 0,
                                "height_shift_range": 0,
                                "zoom_range": 0,
                                "brightness_range": None}

    def get_image_generators(self):
        raise NotImplementedError

    def _split_files_labels(self, all_files, all_labels):
        # Redo the split to obtain train/val/test sets. test set is 10% of total, validation is 9% of total (10% of remaining).
        train_val_files, test_files, train_val_labels, test_labels = train_test_split(all_files, all_labels,
                                                                                      test_size=self.test_size,
                                                                                      random_state=0,
                                                                                      stratify=all_labels)
        train_files, val_files, train_labels, val_labels = train_test_split(train_val_files, train_val_labels,
                                                                            test_size=self.val_size, random_state=0,
                                                                            stratify=train_val_labels)
        return train_files, train_labels, val_files, val_labels, test_files, test_labels

    def _labels_to_intlabels(self, labels):
        return list(map(lambda label_name: self.names_to_ints[label_name], labels))


class StanfordLoader(DataLoader):
    def __init__(self, data_params):
        super().__init__(data_params)
        self.names_to_ints = {'applauding': 0, 'climbing': 1, 'drinking': 2, 'jumping': 3, 'pouring_liquid': 4,
                              'riding_a_bike': 5, 'riding_a_horse': 6,
                              'running': 7, 'shooting_an_arrow': 8, 'smoking': 9, 'throwing_frisby': 10,
                              'waving_hands': 11}
        self.keep_stanford40 = ["applauding", "climbing", "drinking", "jumping", "pouring_liquid", "riding_a_bike",
                                "riding_a_horse",
                                "running", "shooting_an_arrow", "smoking", "throwing_frisby", "waving_hands"]
        self.data_directory = "./data/JPEGImages"

    def get_image_generators(self, use_data_augmentation = True):
        train_files, train_labels, val_files, val_labels, test_files, test_labels = self._get_im_files_labels()
        if self.print_info:
            self._print_info(train_labels, val_labels, test_labels)
        train_generator = self._files_labels_to_datagen(train_files, train_labels, use_data_augmentation= use_data_augmentation)
        val_generator = self._files_labels_to_datagen(val_files, val_labels, use_data_augmentation=False)
        test_generator = self._files_labels_to_datagen(test_files, test_labels, use_data_augmentation=False)
        return train_generator, val_generator, test_generator

    def _files_labels_to_datagen(self, files, labels, use_data_augmentation):
        if use_data_augmentation:
            aug_params = self.train_augmentation
        else:
            aug_params = self.no_augmentation
        int_labels = self._labels_to_intlabels(labels)
        df = pd.DataFrame(list(zip(files, int_labels)),
                          columns=['file_name', 'label'])
        datagen = ImageDataGenerator(preprocessing_function=self.rgb_to_bgr,
                                     rescale=1. / 255,
                                     rotation_range=aug_params["rotation_range"],
                                     horizontal_flip=aug_params["horizontal_flip"],
                                     width_shift_range=aug_params["width_shift_range"],
                                     height_shift_range=aug_params["height_shift_range"],
                                     brightness_range=aug_params["brightness_range"],
                                     zoom_range=aug_params["zoom_range"]
                                     )
        generator = datagen.flow_from_dataframe(dataframe=df, directory=self.data_directory, x_col="file_name",
                                                y_col="label",
                                                class_mode="raw", target_size=self.image_size,
                                                batch_size=self.batch_size,
                                                shuffle=True)
        return generator

    def rgb_to_bgr(self, img):
        return img[..., ::-1]

    def _get_im_files_labels(self):
        with open('data/ImageSplits/train.txt', 'r') as f:
            # We won't use these splits but split them ourselves
            train_files = [file_name for file_name in list(map(str.strip, f.readlines())) if
                           '_'.join(file_name.split('_')[:-1]) in self.keep_stanford40]
            train_labels = ['_'.join(name.split('_')[:-1]) for name in train_files]

        with open('data/ImageSplits/test.txt', 'r') as f:
            # We won't use these splits but split them ourselves
            test_files = [file_name for file_name in list(map(str.strip, f.readlines())) if
                          '_'.join(file_name.split('_')[:-1]) in self.keep_stanford40]
            test_labels = ['_'.join(name.split('_')[:-1]) for name in test_files]

        all_files = train_files + test_files
        all_labels = train_labels + test_labels
        return self._split_files_labels(all_files, all_labels)

    def _print_info(self, train_labels, val_labels, test_labels):
        print(f'Train Distribution:{list(Counter(sorted(train_labels)).items())}\n')
        print(f'Validation Distribution:{list(Counter(sorted(val_labels)).items())}\n')
        print(f'Test Distribution:{list(Counter(sorted(test_labels)).items())}\n')
        action_categories = sorted(list(set(train_labels)))
        print(f'Action categories ({len(action_categories)}):\n{action_categories}')


class Hmdb51Loader(DataLoader):
    def __init__(self, data_params):
        super().__init__(data_params)
        self.names_to_ints = {'clap': 0, 'climb': 1, 'drink': 2, 'jump': 3, 'pour': 4, 'ride_bike': 5, 'ride_horse': 6,
                              'run': 7, 'shoot_bow': 8, 'smoke': 9, 'throw': 10, 'wave': 11}
        self.keep_hmdb51 = ["clap", "climb", "drink", "jump", "pour", "ride_bike", "ride_horse",
                            "run", "shoot_bow", "smoke", "throw", "wave"]

    def get_optical_flow_generators(self, shuffle =True):
        flow_directories = ["data/hmdb51_train_flow", "data/hmdb51_val_flow", "data/hmdb51_test_flow"]
        return self._get_generators_from_directories(flow_directories, use_data_aug=[False, False, False],
                                                     scale_factor=1, shuffle =shuffle)

    def get_image_generators(self, use_data_augmentation= True, shuffle=True):
        image_directories = ["data/hmdb51_train_images", "data/hmdb51_val_images", "data/hmdb51_test_images"]
        if use_data_augmentation:
            return self._get_generators_from_directories(image_directories, use_data_aug=[True, False, False],
                                                     scale_factor=1. / 255, shuffle=shuffle)
        else:
            return self._get_generators_from_directories(image_directories, use_data_aug=[False, False, False],
                                                     scale_factor=1. / 255, shuffle= shuffle)

    def _get_generators_from_directories(self, directories, use_data_aug, scale_factor, shuffle):
        generators = []
        for i in range(len(directories)):
            files = os.listdir(directories[i])
            int_labels = np.array([self.extract_int_label(f) for f in files])
            generators.append(
                self._files_labels_to_datagen(files, int_labels, directories[i], use_data_aug[i], scale_factor, shuffle))
        return generators

    def extract_int_label(self, string):
        match = re.search(r'\d{1,2}', string)
        return int(match.group())

    def _files_labels_to_datagen(self, files, int_labels, directory, use_data_aug, scale_factor, shuffle):
        df = pd.DataFrame(list(zip(files, int_labels)),
                          columns=['file_name', 'label'])
        generator = CustomDataGen(dataframe=df, directory=directory, x_col="file_name",
                                  y_col="label",
                                  input_size=self.image_size,
                                  batch_size=self.batch_size,
                                  scale_factor=scale_factor,
                                  aug_params=self.train_augmentation,
                                  use_data_aug=use_data_aug,
                                  shuffle=shuffle)
        return generator

    '''
    THE FOLLOWING METHODS ARE FOR CREATING AND SAVING NEW DATASETS. 
    '''

    def create_middle_frame_dataset(self):
        train_files, train_labels, val_files, val_labels, test_files, test_labels = self._get_video_files_names()
        self._save_middle_frames(train_files, train_labels, './data/hmdb51_train_images')
        self._save_middle_frames(val_files, val_labels, './data/hmdb51_val_images')
        self._save_middle_frames(test_files, test_labels, './data/hmdb51_test_images')

    def create_optical_flow_dataset(self, opt_flow_frames=16):
        train_files, train_labels, val_files, val_labels, test_files, test_labels = self._get_video_files_names()
        self._save_optical_flow(train_files, train_labels, opt_flow_frames, './data/hmdb51_train_flow')
        self._save_optical_flow(val_files, val_labels, opt_flow_frames, './data/hmdb51_val_flow')
        self._save_optical_flow(test_files, test_labels, opt_flow_frames, './data/hmdb51_test_flow')
        return

    def _get_video_files_names(self):
        all_files, all_labels = [], []
        split_pattern_name = f"*test_split1.txt"
        split_pattern_path = os.path.join('data\\testTrainMulti_7030_splits', split_pattern_name)
        annotation_paths = glob.glob(split_pattern_path)
        for filepath in annotation_paths:
            class_name = '_'.join(filepath.split('\\')[-1].split('_')[:-2])
            if class_name not in self.keep_hmdb51:
                continue  # skipping the classes that we won't use.
            with open(filepath) as fid:
                lines = fid.readlines()
            for line in lines:
                video_filename, _ = line.split()
                all_files.append(video_filename)
                all_labels.append(class_name)
        return self._split_files_labels(all_files, all_labels)

    def _save_middle_frames(self, files, labels, path):
        if not os.path.isdir(path):
            os.mkdir(path)
        int_labels = self._labels_to_intlabels(labels)
        label_counts = {"clap": 0, "climb": 0, "drink": 0, "jump": 0, "pour": 0, "ride_bike": 0, "ride_horse": 0,
                        "run": 0, "shoot_bow": 0, "smoke": 0, "throw": 0, "wave": 0}
        for i in range(len(files)):
            video_name = files[i]
            video = cv.VideoCapture("./data/hmdb51_org/" + str(labels[i]) + "/" + video_name)
            frame_count = int(video.get(cv.CAP_PROP_FRAME_COUNT))
            video.set(cv.CAP_PROP_POS_FRAMES, frame_count // 2)
            ret, frame = video.read()
            if ret:
                np.save(path + "/" + str(int_labels[i]) + str(labels[i]) + "_" + str(label_counts[labels[i]]) + ".npy",
                        frame)
                label_counts[labels[i]] += 1
            if not ret:
                raise Exception("not every video frame returns image")

    def _save_optical_flow(self, files, labels, opt_flow_frames, path):
        if not os.path.isdir(path):
            os.mkdir(path)
        int_labels = self._labels_to_intlabels(labels)
        label_counts = {"clap": 0, "climb": 0, "drink": 0, "jump": 0, "pour": 0, "ride_bike": 0, "ride_horse": 0,
                        "run": 0, "shoot_bow": 0, "smoke": 0, "throw": 0, "wave": 0}
        for i in range(len(files)):
            video_name = files[i]
            video = cv.VideoCapture("./data/hmdb51_org/" + str(labels[i]) + "/" + video_name)
            frame_count = int(video.get(cv.CAP_PROP_FRAME_COUNT))
            frame_numbers = np.linspace(1, frame_count - 2, opt_flow_frames + 1).astype(int)
            stack_of_frames = []
            for j in range(len(frame_numbers)):
                video.set(cv.CAP_PROP_POS_FRAMES, frame_numbers[j])
                ret, frame = video.read()
                if not ret:
                    raise Exception("not every video frame returns image")
                else:
                    stack_of_frames.append(frame)
            optical_flow_stack = self._calculate_optical_flow(stack_of_frames)
            np.save(path + "/" + str(int_labels[i]) + str(labels[i]) + "_" + "flow_" + str(
                label_counts[labels[i]]) + ".npy",
                    optical_flow_stack)
            label_counts[labels[i]] += 1

    def _calculate_optical_flow(self, stack_of_frames):
        optical_flow_stack = []
        prev_fr = cv.cvtColor(stack_of_frames[0], cv.COLOR_BGR2GRAY)
        for i in range(1, len(stack_of_frames)):
            next_fr = cv.cvtColor(stack_of_frames[i], cv.COLOR_BGR2GRAY)
            flow = cv.calcOpticalFlowFarneback(prev_fr, next_fr, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            if len(optical_flow_stack) == 0:
                optical_flow_stack = flow
            else:
                optical_flow_stack = np.concatenate((optical_flow_stack, flow), axis=-1)
            prev_fr = next_fr
        return optical_flow_stack


'''
CUSTOM DATA GEN, for loading in numpy arrays. Inspired by 
https://medium.com/analytics-vidhya/write-your-own-custom-data-generator-for-tensorflow-keras-1252b64e41c3
'''


class CustomDataGen(tf.keras.utils.Sequence):
    def __init__(self, dataframe, x_col, y_col,
                 batch_size,
                 input_size,
                 directory,
                 scale_factor,
                 aug_params,
                 use_data_aug,
                 shuffle=True
                 ):
        self.df = dataframe.copy()
        self.x_col = x_col
        self.y_col = y_col
        self.batch_size = batch_size
        self.input_size = input_size
        self.directory = directory
        self.shuffle = shuffle
        self.n = len(self.df)
        self.current_index = 0
        self.scale_factor = scale_factor
        self.use_data_aug = use_data_aug
        if use_data_aug:
            self.datagen = ImageDataGenerator( #No brightness range transform because weird behaviour with rgb and bgr representation of data
                rotation_range=aug_params["rotation_range"],
                horizontal_flip=aug_params["horizontal_flip"],
                width_shift_range=aug_params["width_shift_range"],
                height_shift_range=aug_params["height_shift_range"],
                zoom_range=aug_params["zoom_range"],
            )

    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)

    def __get_input(self, path, target_size):
        arr = np.load(os.path.join(self.directory, path))
        image_arr = tf.image.resize(arr, (target_size[0], target_size[1])).numpy() * self.scale_factor
        if self.use_data_aug:
            image_arr = self.datagen.random_transform(image_arr)
        return image_arr

    def __get_data(self, batches):
        # Generates data containing batch_size samples
        path_batch = batches[self.x_col]

        label_batch = batches[self.y_col]

        x_batch = np.asarray([self.__get_input(path, self.input_size) for path in path_batch])

        y_batch = np.asarray(label_batch)

        return x_batch, y_batch

    def __getitem__(self, index):
        batches = self.df[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__get_data(batches)
        return X, y

    def __len__(self):
        return self.n // self.batch_size

    def __next__(self):
        if self.current_index >= len(self):
            self.current_index = 0
        result = self.__getitem__(self.current_index)
        self.current_index += 1
        return result


class DualDataGen(tf.keras.utils.Sequence):
    def __init__(self, im_datagen, flow_datagen):
        self.im_datagen = im_datagen
        self.flow_datagen = flow_datagen
        self.current_index = 0

    def __getitem__(self, index):
        im_batch = self.im_datagen.__next__()
        flow_batch = self.flow_datagen.__next__()
        X = [im_batch[0], flow_batch[0]]
        if (im_batch[1] - flow_batch[1]).any():
            raise Exception("image and flow labels are not equal")
        else:
            y = im_batch[1]
            return X, y

    def __len__(self):
        return self.im_datagen.__len__()

    def __next__(self):
        if self.current_index >= len(self):
            self.current_index = 0
        result = self.__getitem__(self.current_index)
        self.current_index += 1
        return result