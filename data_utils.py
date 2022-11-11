import torch
import pandas as pd
import os

from torchvision import transforms
from PIL import Image


LABELS = ['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion']

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, path, transform=None):
        self.path = os.path.join(*path.split('/')[:-1])
        self.data = pd.read_csv(path)
        if not transform:
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.54], [0.26]),
            ])
        self.transform = transform
        
    def __getitem__(self, idx):
        classes = torch.tensor(self.data.loc[idx, LABELS].tolist()).float()
        img_path = self.data.loc[idx, 'Image']

        img = Image.open(os.path.join(self.path, img_path)).convert('RGB')
        img = self.transform(img)
        return img, classes
    
    def __len__(self):
        return len(self.data)

def get_transforms(mode):
    transforms_dict = dict(
        train=transforms.Compose([
            transforms.CenterCrop(320),
            transforms.RandomAffine(degrees=0, scale=(0.8, 1.1)),
            transforms.RandomPerspective(0.3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.54], [0.26]),
        ]),
        val=transforms.Compose([
            transforms.CenterCrop(320),
            transforms.ToTensor(),
            transforms.Normalize([0.54], [0.26]),
        ]),
    )

    return transforms_dict[mode]