from PIL import Image
import os 
from torch.utils.data import Dataset
import torchvision.transforms as transforms 
from torchvision.transforms import InterpolationMode
import torch

transform = transforms.Compose([ 
    transforms.Resize((100,100),interpolation=InterpolationMode.BILINEAR),
    transforms.PILToTensor(),
])
class CatDogData(Dataset):
    def __init__(self, path):
        self.path = path
        self.cats = os.listdir(path+'cat/')
        self.dogs = os.listdir(path+'dog/')
        self.size = len(self.cats) + len(self.dogs)
        self.images = self.cats + self.dogs
        self.labels = ['cat' for i in self.cats] + ['dog' for i in self.dogs]
    def __len__(self):
        return self.size
    def __getitem__(self, idx):
        label = self.labels[idx]
        img = Image.open(self.path + label + '/' + self.images[idx])
        if(label == 'cat'): label = 0
        else: label = 1
        img = transform(img)
        img = img.type(torch.float32)/255
        return (img, label)