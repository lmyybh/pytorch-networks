import os
from PIL import Image
from torch.utils.data import Dataset

class FFHQDataset(Dataset):
    def __init__(self, data_dir='/data1/cgl/dataset/face/seeprettyface_yellow_face_128/thumbnails128x128', num=None, transforms=None):
        self.paths = [os.path.join(data_dir, n) for n in os.listdir(data_dir)]
        if num is not None and num < len(self.paths):
            self.paths = self.paths[:num]
        self.transforms = transforms
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx])
        if self.transforms is not None:
            img = self.transforms(img)
        return img

    
class DogDataset(Dataset):
    def __init__(self, data_dir='/data1/cgl/dataset/generative-dog-images/square_crop_dogs', num=None, transforms=None):
        self.paths = [os.path.join(data_dir, file) for file in os.listdir(data_dir)]
        if num is not None and num < len(self.paths):
            self.paths = self.paths[:num]
        self.transforms = transforms
        
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx])
        if self.transforms is not None:
            img = self.transforms(img)
        return img

class PokemonDataset(Dataset):
    def __init__(self, data_dir='/data1/cgl/dataset/pokemon_mugshot', num=None, transforms=None):
        self.paths = [os.path.join(data_dir, file) for file in os.listdir(data_dir)]
        if num is not None and num < len(self.paths):
            self.paths = self.paths[:num]
        self.transforms = transforms
        
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx])
        if self.transforms is not None:
            img = self.transforms(img)
        return img
    