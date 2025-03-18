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
from skimage import io
from skimage.transform import resize
import pickle

face_frames = 32
OpenFace_frames = 64
#OpenFace_frames = 96

seq = iaa.Sequential([
    iaa.Add(value=(-40,40), per_channel=True), # Add color 
    iaa.GammaContrast(gamma=(0.5,1.5)) # GammaContrast with a gamma of 0.5 to 1.5
])
"""
this dataloader uses the extracted features. How to extract features please refer to Load_RLT_face_audio_wave_OpenFace_Affect.py.
Note that wave feature is not used here. If you want to use wave features, add network for it.

"""


class Normaliztion(object):
    """
        same as mxnet, normalize into [-1, 1]
        image = (image - 127.5)/128
    """
    def __call__(self, sample):
        video_x, audio_x, OpenFace_x, DD_label, x_affect  = sample['video_x'], sample['audio_x'], sample['OpenFace_x'], sample['DD_label'], sample['x_affect'] 
        new_video_x = (video_x - 127.5)/128     # [-1,1]
        new_audio_x = (audio_x - 127.5)/128     # [-1,1]
        new_OpenFace_x = OpenFace_x
        #new_OpenFace_x = (OpenFace_x*255 - 127.5)/128     # [-1,1], OpenFace_x --> Original [0,1]
        return {'video_x': new_video_x, 'audio_x': new_audio_x, 'OpenFace_x': new_OpenFace_x, 'DD_label': DD_label, 'x_affect': x_affect}



class RandomHorizontalFlip(object):
    """Horizontally flip the given Image randomly with a probability of 0.5."""
    def __call__(self, sample):
        video_x, audio_x, OpenFace_x, DD_label, x_affect  = sample['video_x'], sample['audio_x'], sample['OpenFace_x'], sample['DD_label'], sample['x_affect'] 
        
        new_video_x = np.zeros((face_frames, 224, 224, 3))

        p = random.random()
        if p < 0.5:
            for i in range(face_frames):
                # video 
                image = video_x[i, :, :, :]
                image = cv2.flip(image, 1)
                new_video_x[i, :, :, :] = image

                
            return {'video_x': new_video_x, 'audio_x': audio_x, 'OpenFace_x': OpenFace_x, 'DD_label': DD_label, 'x_affect': x_affect }
        else:
            #print('no Flip')
            return {'video_x': video_x, 'audio_x': audio_x, 'OpenFace_x': OpenFace_x, 'DD_label': DD_label, 'x_affect': x_affect }



class ToTensor(object):
    """
        Convert ndarrays in sample to Tensors.
        process only one batch every time
    """

    def __call__(self, sample):
        video_x, audio_x, OpenFace_x, DD_label, x_affect  = sample['video_x'], sample['audio_x'], sample['OpenFace_x'], sample['DD_label'], sample['x_affect'] 
        
        # swap color axis because
        # numpy image: (batch_size) x T x H x W x C
        # torch image: (batch_size) x C X T x H X W
        video_x = video_x.transpose((3, 0, 1, 2))
        video_x = np.array(video_x)
        
        audio_x = audio_x.transpose((2, 0, 1))
        audio_x = np.array(audio_x)
        
        # numpy image: (batch_size) x T x C
        # torch image: (batch_size) x C X T
        OpenFace_x = OpenFace_x.transpose((1, 0))
        OpenFace_x = np.array(OpenFace_x)
        
        x_affect = x_affect.transpose((1, 0))
        x_affect = np.array(x_affect)
                        
        DD_label_np = np.array([0],dtype=np.long)
        DD_label_np[0] = DD_label
        
        return {'video_x': torch.from_numpy(video_x.astype(np.float)).float(), 'audio_x': torch.from_numpy(audio_x.astype(np.float)).float(), 'OpenFace_x': torch.from_numpy(OpenFace_x.astype(np.float)).float(), 'DD_label': torch.from_numpy(DD_label_np.astype(np.long)).long(), 'x_affect': torch.from_numpy(x_affect.astype(np.float)).float()}


class ToTensor_test(object):
    """
        Convert ndarrays in sample to Tensors.
        process only one batch every time
    """

    def __call__(self, sample):
        video_x, audio_x, OpenFace_x, DD_label, x_affect  = sample['video_x'], sample['audio_x'], sample['OpenFace_x'], sample['DD_label'], sample['x_affect'] 
        
        # swap color axis because
        # numpy image: (batch_size) x Clip x T x H x W x C
        # torch image: (batch_size) x Clip x C X T x H X W
        video_x = video_x.transpose((0, 4, 1, 2, 3))
        video_x = np.array(video_x)
        
        audio_x = audio_x.transpose((0, 3, 1, 2))
        audio_x = np.array(audio_x)
        
        # numpy image: (batch_size) x Clip x T x C
        # torch image: (batch_size) x Clip x C X T
        OpenFace_x = OpenFace_x.transpose((0, 2, 1))
        OpenFace_x = np.array(OpenFace_x)
        
        x_affect = x_affect.transpose((0, 2, 1))
        x_affect = np.array(x_affect)
                        
        DD_label_np = np.array([0],dtype=np.long)
        DD_label_np[0] = DD_label
        
        
        return {'video_x': torch.from_numpy(video_x.astype(np.float)).float(), 'audio_x': torch.from_numpy(audio_x.astype(np.float)).float(), 'OpenFace_x': torch.from_numpy(OpenFace_x.astype(np.float)).float(), 'DD_label': torch.from_numpy(DD_label_np.astype(np.long)).long(), 'x_affect': torch.from_numpy(x_affect.astype(np.float)).float()}





class RLT_train(Dataset):

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

class RLT_test(Dataset):

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
