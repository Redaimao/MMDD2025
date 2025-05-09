import pickle
from torch.utils.data import Dataset


class DOLOS_train(Dataset):
    def __init__(self, list_dir, transform=None):
        """
        Args:
            list_dir  (str): Path to 'train.pkl'.
            transform (callable, optional): A function/transform that takes
                in a sample dict and returns a transformed version.
        """
        with open(list_dir, 'rb') as f:
            # Deserialize list of sample dicts
            self.data_list = pickle.load(f)
        self.transform = transform

    def __len__(self):
        # Total number of training samples
        return len(self.data_list)

    def __getitem__(self, idx):
        """
        Returns:
            dict with keys:
                'video_x'     : np.ndarray of shape (32,224,224,3)
                'x_affect'    : np.ndarray of shape (64,7)
                'audio_x'     : np.ndarray of shape (224,224,3)
                'OpenFace_x'  : np.ndarray of shape (64,43)
                'audio_wave'  : torch.Tensor of shape (1, T)
                'DD_label'    : np.ndarray scalar {0,1}
                'videoname'   : str
        """
        sample = self.data_list[idx]

        out = {
            'video_x': sample['video_x'],
            'x_affect': sample['x_affect'],
            'audio_x': sample['audio_x'],
            'OpenFace_x': sample['OpenFace_x'],
            'audio_wave': sample['audio_wave'],
            'DD_label': sample['DD_label'],
            'videoname': sample.get('videoname', None)
        }

        if self.transform:
            out = self.transform(out)

        return out


class DOLOS_test(Dataset):
    def __init__(self, list_dir, transform=None):
        """
        Same as training class, but loads 'test.pkl'.
        """
        with open(list_dir, 'rb') as f:
            self.data_list = pickle.load(f)
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        """
        Same output dict as train, but note that your stored
        test samples already have the extra leading dims.
        """
        sample = self.data_list[idx]

        out = {
            'video_x': sample['video_x'],  # shape (1,32,224,224,3)
            'x_affect': sample['x_affect'],  # shape (1,64,7)
            'audio_x': sample['audio_x'],  # shape (1,224,224,3)
            'OpenFace_x': sample['OpenFace_x'],  # shape (1,64,43)
            'audio_wave': sample['audio_wave'],  # shape (1, T)
            'DD_label': sample['DD_label'],  # 0-D np.array
            'videoname': sample.get('videoname', None)
        }

        if self.transform:
            out = self.transform(out)

        return out
