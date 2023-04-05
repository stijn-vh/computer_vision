"""## Read the train and test splits, combine them and make better splits to help training networks easier."""

from collections import Counter
from sklearn.model_selection import train_test_split
import os
import torchvision
import torch.nn.functional as F
from torchvision import transforms
import torch.optim as optim
from torch.utils.data import random_split, DataLoader

import torch
import cv2 as cv

if __name__ == '__main__':

    keep_stanford40 = ["applauding", "climbing", "drinking", "jumping", "pouring_liquid", "riding_a_bike",
                       "riding_a_horse",
                       "running", "shooting_an_arrow", "smoking", "throwing_frisby", "waving_hands"]

    with open('data/ImageSplits/train.txt', 'r') as f:
        # We won't use these splits but split them ourselves
        train_files = [file_name for file_name in list(map(str.strip, f.readlines())) if
                       '_'.join(file_name.split('_')[:-1]) in keep_stanford40]
        train_labels = ['_'.join(name.split('_')[:-1]) for name in train_files]

    with open('data/ImageSplits/test.txt', 'r') as f:
        # We won't use these splits but split them ourselves
        test_files = [file_name for file_name in list(map(str.strip, f.readlines())) if
                      '_'.join(file_name.split('_')[:-1]) in keep_stanford40]
        test_labels = ['_'.join(name.split('_')[:-1]) for name in test_files]

    # Combine the splits and split for keeping more images in the training set than the test set.
    all_files = train_files + test_files
    all_labels = train_labels + test_labels
    train_files, test_files = train_test_split(all_files, test_size=0.1, random_state=0, stratify=all_labels)
    train_labels = ['_'.join(name.split('_')[:-1]) for name in train_files]
    test_labels = ['_'.join(name.split('_')[:-1]) for name in test_files]

    print(f'Train files ({len(train_files)}):\n\t{train_files}')
    print(f'Train labels ({len(train_labels)}):\n\t{train_labels}\n' \
          f'Train Distribution:{list(Counter(sorted(train_labels)).items())}\n')
    print(f'Test files ({len(test_files)}):\n\t{test_files}')
    print(f'Test labels ({len(test_labels)}):\n\t{test_labels}\n' \
          f'Test Distribution:{list(Counter(sorted(test_labels)).items())}\n')
    action_categories = sorted(list(set(train_labels)))
    print(f'Action categories ({len(action_categories)}):\n{action_categories}')

    # """### Visualize a photo from the training files and also print its label"""
    # image_no = 234  # change this to a number between [0, 1200] and you can see a different training image
    # img = cv.imread(f'data/JPEGImages/{train_files[image_no]}')
    # cv.imshow(f'An image with the label - {train_labels[image_no]}', img)
    # cv.waitKey(1000)

    keep_hmdb51 = ["clap", "climb", "drink", "jump", "pour", "ride_bike", "ride_horse",
                   "run", "shoot_bow", "smoke", "throw", "wave"]

    """# Load data into dataloaders with necessary transforms"""
    torch.manual_seed(97)
    num_frames = 16
    clip_steps = 2
    batch_size = 16

    # torchvision.transforms.ToFloatTensorInZeroOne(),

    transform = transforms.Compose([
                                    transforms.Resize((200, 200)),
                                    transforms.RandomCrop((172, 172))])
    transform_test = transforms.Compose([
                                         transforms.Resize((200, 200)),
                                         transforms.CenterCrop((172, 172))])

    hmdb51_train = torchvision.datasets.HMDB51('data/hmdb51_org', 'data/testTrainMulti_7030_splits/', num_frames,
                                               frame_rate=5,
                                               step_between_clips=clip_steps, fold=1, train=True,
                                               transform=transform, num_workers=2)

    hmdb51_test = torchvision.datasets.HMDB51('data/hmdb51_org', 'data/testTrainMulti_7030_splits/', num_frames,
                                              frame_rate=5,
                                              step_between_clips=clip_steps, fold=1, train=False,
                                              transform=transform_test, num_workers=2)

    train_loader = DataLoader(hmdb51_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(hmdb51_test, batch_size=batch_size, shuffle=False)

    """# Let's print the data shape with batch size 16 and 16 frames."""

    for data, _, labels in train_loader:
        print(data.shape)  # 16-batch size, 3-channels, 16-frames, 172x172-crop
        print(labels)  # 12 classes [0-11]
        break
