import os
import pickle
import numpy as np
import pandas as pd
from PIL import Image
import torch

torch
import torchaudio
from torch.utils.data import Dataset
from torchaudio.functional import resample


class TrainDataset(Dataset):
    def __init__(self,
                 groundtruth_path: str,
                 frames_root: str,
                 mel_root: str,
                 affect_root: str,
                 openface_root: str,
                 audio_root: str):
        # Read the two-column groundtruth file (filename, label)
        df = pd.read_csv(groundtruth_path, sep=r'\s+', header=None, names=['name', 'label'])
        self.samples = list(df.itertuples(index=False, name=None))
        self.frames_root = frames_root
        self.mel_root = mel_root
        self.affect_root = affect_root
        self.openface_root = openface_root
        self.audio_root = audio_root

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        name, label = self.samples[idx]
        print("sample :{}, name: {}\n".format(idx, name))

        # --- Video frames ---
        frame_dir = os.path.join(self.frames_root, name)
        all_frames = sorted(os.listdir(frame_dir))
        indices = np.linspace(0, len(all_frames) - 1, num=32, dtype=int)
        video_x = []
        for i in indices:
            img = Image.open(os.path.join(frame_dir, all_frames[i]))
            img = img.resize((224, 224), Image.BICUBIC)
            video_x.append(np.array(img))
        video_x = np.stack(video_x, axis=0)  # (32,224,224,3)

        # --- Mel-spectrogram image ---
        mel_img = Image.open(os.path.join(self.mel_root, f"{name}.png"))
        audio_x = np.array(mel_img)  # (224,224,3)

        # --- Affect features ---
        x_affect = np.load(os.path.join(self.affect_root, f"{name}.npy"))  # (64,7)

        # --- OpenFace features ---
        df_of = pd.read_csv(os.path.join(self.openface_root, f"{name}.csv"))
        OpenFace_x = df_of.values  # (64,43)

        # --- Audio waveform ---
        wav_path = os.path.join(self.audio_root, name[:3], f"{name}.wav")
        waveform, sr = torchaudio.load(wav_path)
        if sr != 16000:
            waveform = resample(waveform, orig_freq=sr, new_freq=16000)
        audio_wave = waveform[0].unsqueeze(0)

        info = {
            'videoname': name,
            'DD_label': np.array(label),
            'video_x': video_x,
            'audio_x': audio_x,
            'x_affect': x_affect,
            'OpenFace_x': OpenFace_x,
            'audio_wave': audio_wave
        }
        # print("video_x\n", info['video_x'].shape)  # (32, 224, 224, 3)     # (1, 32, 224, 224, 3)
        # print("x_affect\n", info['x_affect'].shape)  # (64, 7)             #(1, 64, 7)
        # print("audio_x\n", info['audio_x'].shape)  # (224, 224, 3)        #(1, 224, 224, 3)
        # print("OpenFace_x\n", info['OpenFace_x'].shape)  # (64, 43)        # (1, 64, 43)
        # print("DD_label\n", info['DD_label'].shape, info['DD_label'])  # () 0
        # print("audio_wave\n", info['audio_wave'].shape)  # torch.Size([1, 162880])
        # print("videoname\n", info['videoname'])  # User_1/run_4

        return info


class TestDataset(TrainDataset):
    def __init__(self,
                 groundtruth_path: str,
                 frames_root: str,
                 mel_root: str,
                 affect_root: str,
                 openface_root: str,
                 audio_root: str):
        super().__init__(
            groundtruth_path, frames_root, mel_root,
            affect_root, openface_root, audio_root
        )

    def __getitem__(self, idx):
        # Load all features as in TrainDataset
        sample = super().__getitem__(idx)
        # Remove label since test set should not expose it
        sample.pop('DD_label', None)

        # Expand dims for test samples
        sample['video_x'] = np.expand_dims(sample['video_x'], axis=0)
        sample['audio_x'] = np.expand_dims(sample['audio_x'], axis=0)
        sample['x_affect'] = np.expand_dims(sample['x_affect'], axis=0)
        sample['OpenFace_x'] = np.expand_dims(sample['OpenFace_x'], axis=0)

        return sample


# Example of saving the prepared test dataset to a pickle file
if __name__ == "__main__":
    # Example: Saving the training data
    train_dataset = TrainDataset(
        groundtruth_path="MDPE/train_balanced_l493_t492_GT.txt",
        frames_root="/face_frames_retina",
        mel_root="/mel_spectrogram",
        affect_root="/affect_features",
        openface_root="/openface_43",
        audio_root="MDPE/raw_audio"
    )
    # data = train_dataset[100]
    # Materialize all samples into memory (be mindful of RAM!)
    print(len(train_dataset))
    train_list = [train_dataset[i] for i in range(len(train_dataset))]
    # #
    # # # Serialize to train.pkl
    # with open("/MDPE/train_balanced_l493_t492_GT.pkl", "wb") as f:
    #     pickle.dump(train_list, f)  # :contentReference[oaicite:18]{index=18}
    # print("finished train data features\n")



    # test_dataset = TestDataset(
    #     groundtruth_path="/test_balanced_l127_t124_GT.txt",
    #     frames_root="MDPE/face_frames_retina",
    #     mel_root="MDPE/mel_spectrogram",
    #     affect_root="MDPE/affect_features",
    #     openface_root="MDPE/openface_43",
    #     audio_root="//MDPE/raw_audio"
    # )
    # test_list = [test_dataset[i] for i in range(len(test_dataset))]
    # with open("MDPE/test_balanced_l127_t124_GT.pkl", "wb") as f:
    #     pickle.dump(test_list, f)
    # print("finished test data features\n")
    #
