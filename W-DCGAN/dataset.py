import os
from torch.utils.data import Dataset
from PIL import Image

class DogDataset(Dataset):
    def __init__(self, data_dir, mode='train', ratio=0.8, transforms=None):
        assert mode in ['train', 'test'], 'mode must be train or test'
        assert 0.1 <= ratio <= 0.9, 'ratio must be between 0.1~0.9'
        
        files = [os.path.join(data_dir, file) for file in os.listdir(data_dir)]
        split_idx = int(len(files) * ratio)
        self.files = files[:split_idx] if mode == 'train' else files[split_idx:]
        self.transforms = transforms
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        img = Image.open(self.files[index])
        if self.transforms is not None:
            img = self.transforms(img)
        return img
    