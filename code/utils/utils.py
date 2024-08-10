import os
import random
import torch
import numpy as np
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


# set_seed() function is used to set the seed for reproducibility
def set_seed(seed):
    """
    Set the seed for reproducibility
    input: seed value
    """
    random.seed(seed)               
    np.random.seed(seed)            
    torch.manual_seed(seed)         
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# LoadDataset class is used to load the dataset for training
class LoadDataset():
    """
    Load the dataset for training
    input: dataset_path, batch_size
    output: dataloader
    """
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

# plot_training_curve() function is used to plot the training curve
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

# RandomErasing class is used to randomly erase a rectangle from the image
class RandomErasing:
    """
    Randomly erase a rectangle from the image
    input: p(probability), value(value of the pixel)
    output: img
    """
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

