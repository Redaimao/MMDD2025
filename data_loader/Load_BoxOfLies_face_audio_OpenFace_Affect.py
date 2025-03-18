from __future__ import print_function, division
import os
import torch
import pandas as pd
#from skimage import io, transform
import cv2
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pdb
import math
import os 
import imgaug.augmenters as iaa

import pickle

face_frames = 32
OpenFace_frames = 64
#OpenFace_frames = 96

"""
this dataloader uses the extracted features. How to extract features please refer to Load_BoxOfLies_face_audio_wave_OpenFace_Affect.py.
Note that wave feature is not used here. If you want to use wave features, add network for it.

Note that Box of lies is used to test, not for training. 
"""


class BoxOfLies_train(Dataset):
    """
    Not used.
    """

    def __init__(self, list_dir,  transform=None):
        with open(list_dir, 'rb') as f:
            self.data_list = pickle.load(f)
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    
    def __getitem__(self, idx):
        # return format --> sample = {'video_x': video_x, 'x_affect': x_affect, 'audio_x': audio_x, 'OpenFace_x': OpenFace_x, 'DD_label': DD_label, 'audio_wave': audio_wave, 'videoname': videoname}

        sample = self.data_list[idx]

        if self.transform:
            sample = self.transform(sample)
        return sample

class BoxOfLies_test(Dataset):

    def __init__(self, list_dir,  transform=None):
        with open(list_dir, 'rb') as f:
            self.data_list = pickle.load(f)
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    
    def __getitem__(self, idx):
        # return format --> sample = {'video_x': video_x, 'x_affect': x_affect, 'audio_x': audio_x, 'OpenFace_x': OpenFace_x, 'DD_label': DD_label, 'audio_wave': audio_wave, 'videoname': videoname}

        sample = self.data_list[idx]

        if self.transform:
            sample = self.transform(sample)
        return sample


