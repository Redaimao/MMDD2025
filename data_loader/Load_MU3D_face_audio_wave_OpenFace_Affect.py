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
import torchaudio


face_frames = 32
OpenFace_frames = 64
#OpenFace_frames = 96

seq = iaa.Sequential([
    iaa.Add(value=(-40,40), per_channel=True), # Add color 
    iaa.GammaContrast(gamma=(0.5,1.5)) # GammaContrast with a gamma of 0.5 to 1.5
])







class MU3D_train(Dataset):

    def __init__(self, info_list, root_dir,  transform=None):

        self.landmarks_frame = pd.read_csv(info_list, delimiter=' ', header=None)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    
    def __getitem__(self, idx):
        #print(self.landmarks_frame.iloc[idx, 0])
        videoname = str(self.landmarks_frame.iloc[idx, 0])
        video_path = os.path.join(self.root_dir, videoname)
        DD_label = str(self.landmarks_frame.iloc[idx, 1])
        totalframes  = self.landmarks_frame.iloc[idx, 2]
        clip_num  = self.landmarks_frame.iloc[idx, 3]
        
        video_x, audio_x, OpenFace_x, x_affect, audio_wave = self.get_video_x(video_path, totalframes, clip_num, videoname)
        
        sample = {'video_x': video_x, 'x_affect': x_affect, 'audio_x': audio_x, 'OpenFace_x': OpenFace_x, 'DD_label': DD_label, 'audio_wave': audio_wave}

        if self.transform:
            sample = self.transform(sample)
        return sample
    
    
    def get_video_x(self, video_path, totalframes, clip_num, name):
        
        video_x = np.zeros((face_frames, 224, 224, 3))

        if clip_num==1:
            clip_id = 1
        else:
            clip_id = np.random.randint(1, clip_num)
        clip_length = totalframes//clip_num - 1
        interval = clip_length//face_frames
        
        
        offset = np.random.randint(1, interval)-1
        start_frame = (clip_id-1)*clip_length + 1 + offset
        
        image_id = start_frame
        
 
        for i in range(face_frames):
            s = "%04d" % image_id
            video_name = 'frame_' + s + '.jpg'

            # face video 
            video_path_temp = os.path.join(video_path, video_name)
            
            tmp_image = cv2.imread(video_path_temp)
            
            #tmp_image = cv2.resize(tmp_image, (112, 112), interpolation=cv2.INTER_CUBIC)
            
            video_x[i, :, :, :] = tmp_image  
                        
            image_id += interval 
        
        # Audio  Spectrogram
        audio_x = np.zeros((224, 224, 3))
            
        s2 = "%d" % clip_id
        image_path = '/Deception_datasets/MU3D/MU3D_spectrograms/' + name + '_' + s2 + '.png'
     
        image_x_temp = cv2.imread(image_path)
        audio_x = cv2.resize(image_x_temp, (224, 224))
        audio_x_aug = seq.augment_image(audio_x) 
        
        
        # Audio Waveform
        w2v2_sr=16000
        
        audio_path = '/Deception_datasets/MU3D/M_audio_files/'+ name + '.wav'

        waveform, sample_rate = torchaudio.load(audio_path) # industry standard is 44.1 khz

        # Resample to the model sampling frequency. VERY IMPORTANT STEP!!!
        if sample_rate != w2v2_sr:
            waveform = torchaudio.functional.resample(waveform, sample_rate, w2v2_sr)
        
        audio_wave = waveform[0].unsqueeze(0)
        
        
        
        
        # OpenFace
        OpenFace_x = np.zeros((OpenFace_frames, 43))
        
        offset_OpenFace = np.random.randint(1, clip_length-OpenFace_frames)-1
        start_frame = (clip_id-1)*clip_length + 1 + offset_OpenFace
        
        image_path = '/Deception_datasets/MU3D/mu3d_openface43/' + name + '.csv'
         
        OpenFace_x = pd.read_csv(image_path, header=None)  
        #OpenFace_x = pd.DataFrame(OpenFace_x).loc[(start_frame_OpenFace):(start_frame_OpenFace+OpenFace_frames),1:]  # ingore the first dim
        OpenFace_x = pd.DataFrame(OpenFace_x).loc[(start_frame):(start_frame+OpenFace_frames-1),:]  # ingore the first dim
        OpenFace_x = np.array(OpenFace_x, dtype=np.float64)
        
        
        # 
        # Affect face input according to OpenFace
        # 
        
        # [ expression 5 + 1 arousal + 1 valence ]
        x_affect = np.zeros((OpenFace_frames, 7))
        
        start_frame = (clip_id-1)*clip_length + 1 + offset_OpenFace
        
        image_id = start_frame
        for i in range(OpenFace_frames):
            s = "%04d" % image_id
            video_name = 'frame_' + s + '.npy'

            # face video 
            video_path_temp = os.path.join(video_path, video_name)
            
            affect = np.load(video_path_temp)
            
            x_affect[i, :] = affect  
                        
            image_id += 1
        
        
        return video_x, audio_x, OpenFace_x, x_affect, audio_wave




class MU3D_test(Dataset):

    def __init__(self, info_list, root_dir,  transform=None):

        self.landmarks_frame = pd.read_csv(info_list, delimiter=' ', header=None)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    
    def __getitem__(self, idx):
        #print(self.landmarks_frame.iloc[idx, 0])
        videoname = str(self.landmarks_frame.iloc[idx, 0])
        video_path = os.path.join(self.root_dir, videoname)
        DD_label = str(self.landmarks_frame.iloc[idx, 1])
        totalframes  = self.landmarks_frame.iloc[idx, 2]
        clip_num  = self.landmarks_frame.iloc[idx, 3]
        
        video_x, audio_x, OpenFace_x, x_affect, audio_wave = self.get_video_x(video_path, totalframes, clip_num, videoname)
        
        sample = {'video_x': video_x, 'audio_x': audio_x, 'OpenFace_x': OpenFace_x, 'DD_label': DD_label, 'x_affect': x_affect, 'audio_wave': audio_wave}

        if self.transform:
            sample = self.transform(sample)
        return sample
    
    
    def get_video_x(self, video_path, totalframes, clip_num, name):
        
        video_x = np.zeros((clip_num, face_frames, 224, 224, 3))

        # randomly sampling 'totalframes' from each clip
        #clip_id = np.random.randint(1, 2)
        clip_length = totalframes//clip_num - 1
        interval = clip_length//face_frames
        
        for tt in range(clip_num):
            
            image_id = tt*clip_length + 1 
            
            for i in range(face_frames):
                s = "%04d" % image_id
                video_name = 'frame_' + s + '.jpg'
    
                # face video 
                video_path_temp = os.path.join(video_path, video_name)
                
                tmp_image = cv2.imread(video_path_temp)
                
                #tmp_image = cv2.resize(tmp_image, (112, 112), interpolation=cv2.INTER_CUBIC)
                
                video_x[tt, i, :, :, :] = tmp_image  
                            
                image_id += interval 
        
        # Audio Spectrogram
        audio_x = np.zeros((clip_num, 224, 224, 3))
        
        for tt in range(clip_num):
            
            s2 = "%d" % (tt+1)
            image_path = '/Deception_datasets/MU3D/MU3D_spectrograms/' + name + '_' + s2 + '.png'
    
            image_x_temp = cv2.imread(image_path)
            audio_x[tt, :, :, :] = cv2.resize(image_x_temp, (224, 224))
        
        # Audio Waveform
        w2v2_sr=16000
        
        audio_path = '/Deception_datasets/MU3D/M_audio_files/'+ name + '.wav'

        waveform, sample_rate = torchaudio.load(audio_path) # industry standard is 44.1 khz

        # Resample to the model sampling frequency. VERY IMPORTANT STEP!!!
        if sample_rate != w2v2_sr:
            waveform = torchaudio.functional.resample(waveform, sample_rate, w2v2_sr)
        
        audio_wave = waveform[0].unsqueeze(0)
        
        
        # OpenFace
        OpenFace_x = np.zeros((clip_num, OpenFace_frames, 43))
        
        image_path = '/scratch/project_2004030/Deception_datasets/MU3D/mu3d_openface43/' + name + '.csv'
             
        OpenFace_x_temp = pd.read_csv(image_path, header=None)  
        
        for tt in range(clip_num):
            start_frame = tt*clip_length + 1 
                 
            OpenFace_x_temp2 = pd.DataFrame(OpenFace_x_temp).loc[(start_frame):(start_frame+OpenFace_frames-1),:]  # ingore the first dim
            OpenFace_x_temp2 = np.array(OpenFace_x_temp2, dtype=np.float64)
            OpenFace_x[tt, :, :] = OpenFace_x_temp2
        
        # 
        # Affect face input according to OpenFace
        # 
        
        x_affect = np.zeros((clip_num, OpenFace_frames, 7))
        
        for tt in range(clip_num):
        
            image_id = tt*clip_length + 1 
            
            for i in range(face_frames):
                s = "%04d" % image_id
                video_name = 'frame_' + s + '.npy'
    
                # face video 
                video_path_temp = os.path.join(video_path, video_name)
                
                affect = np.load(video_path_temp)
                
                x_affect[tt, i, :] = affect  
                            
                image_id += 1 
        
        
        return video_x, audio_x, OpenFace_x, x_affect, audio_wave

