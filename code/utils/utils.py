import os
import random
import torch
import numpy as np
from torchvision import datasets, transforms
import matplotlib.pyplot as plt



def set_seed(seed):
    random.seed(seed)               
    np.random.seed(seed)            
    torch.manual_seed(seed)         
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class LoadDataset():
    def __init__(self, dataset_path, batch_size):
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    def load_data(self):
        try:
            data = datasets.ImageFolder(self.dataset_path, transform=self.transform)
            dataloader = torch.utils.data.DataLoader(data, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=8)
            print(f"Number of images loaded: {len(data)}")
            return dataloader
        except Exception as e:
            print(f"Error loading dataset: {e}")


def plot_training_curve(epochs, losses, augmentation_name, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, losses, label='Training Loss')
    plt.title(f'{augmentation_name} Training Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_path, f'{augmentation_name}_training_curve.png'))
    plt.close()

class RandomErasing:
    def __init__(self, p=0.5, value=0):
        self.p = p
        self.value = value
        self.crop_size = (64, 64)

    def __call__(self, img):
        if random.random() < self.p:
            c, h, w = img.size()
            left = random.randint(0, w - self.crop_size[1])
            top = random.randint(0, h - self.crop_size[0])
            img[:, top:top+self.crop_size[0], left:left+self.crop_size[1]] = self.value
        return img

