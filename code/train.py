import argparse
import random
from utils.utils import LoadDataset
from utils.utils import RandomErasing
from utils.utils import set_seed
from simclr.simclr_model import SimCLRTrainer
from byol.byol_model import BYOLTrainer
from moco.moco_model import MoCoTrainer
import torchvision
from torch import nn
from torchvision import datasets, transforms

"""
Training Script
input: model, batch_size, learning_rate, epochs, augmentation, seed
output: trained model
"""

def main():
    parser = argparse.ArgumentParser(description="Training Script")
    parser.add_argument("model", type=str, choices=["simclr", "byol", "moco"], help="Select model type (simclr or byol or moco)")
    parser.add_argument("--batch_size", type=int, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, help="Learning rate for training")
    parser.add_argument("--epochs", type=int, help="Number of epochs for training")
    parser.add_argument("--augmentation", type=str, help="Augmentation strategy")
    parser.add_argument("--seed", type=int, choices=[0, 42, 123], help="Seed value for reproducibility")

    
    args = parser.parse_args()
    
    # Replace the dataset_path with the path to the Pascal VOC dataset on your system
    # Replace the model_path with the path where you want to save the trained model
    dataset_path = f"/home/jovyan/data/pascal_voc/"

    if args.seed == 0:        
        model_path = f"/home/jovyan/models/trained_models/seed_zero/{args.model}/"
        seed = 0
    elif args.seed == 42:
        model_path = f"/home/jovyan/models/trained_models/seed_42/{args.model}/"
        seed = 42
    elif args.seed == 123:
        model_path = f"/home/jovyan/models/trained_models/seed_123/{args.model}/"
        seed = 123
    
    set_seed(seed)
    
    # Load the dataset
    loading_data = LoadDataset(dataset_path, args.batch_size)
    data = loading_data.load_data()

    augmentation_strategy = None
    augmentation_name = ""
    # Select the augmentation strategy
    if args.augmentation == "CenterCrop":
        augmentation_strategy = transforms.CenterCrop(size=(64, 64))
        augmentation_name = "center_cropping"
    
    elif args.augmentation == "RandomCrop":
        augmentation_strategy = transforms.RandomCrop(size=(64, 64))
        augmentation_name = "random_cropping"
    
    elif args.augmentation == "ColorJitter":
        augmentation_strategy = transforms.ColorJitter(
            brightness=random.random(),
            contrast=random.random(),
            saturation=random.random(),
            hue=random.uniform(0.0, 0.5))
        augmentation_name = "color_jitter"
    
    elif args.augmentation == "RandomFlipping":
        augmentation_strategy = transforms.RandomApply([
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomHorizontalFlip(p=0.5)], p=1.0)
        augmentation_name = "random_flipping"
    
    elif args.augmentation == "RandomPerspective":
        augmentation_strategy = transforms.RandomPerspective(distortion_scale=0.5, p=1.0)
        augmentation_name = "random_perspective"

    elif args.augmentation == "RandomRotation":
        augmentation_strategy = transforms.RandomRotation(degrees=(0, 360))
        augmentation_name = "random_rotation"
        
    elif args.augmentation == "RandomGrayscale":
        augmentation_strategy = transforms.RandomGrayscale(p=0.5)
        augmentation_name = "random_grayscale"
        
    elif args.augmentation == "GaussianBlur":
        augmentation_strategy = transforms.GaussianBlur(kernel_size=random.choice([3, 5, 7]), sigma=(0.1, 2.0))
        augmentation_name = "gaussian_blur"
        
    elif args.augmentation == "RandomInvert":
        augmentation_strategy = transforms.RandomInvert(p=0.5)
        augmentation_name = "random_invert"
        
    elif args.augmentation == "RandomErasing":
        augmentation_strategy = RandomErasing(p=0.5)
        augmentation_name = "random_erasing"
        

    # Train the model
    if args.model == "simclr":
        trainer = SimCLRTrainer(data, args.learning_rate, args.epochs, augmentation_strategy, augmentation_name, args.seed)
    elif args.model == "byol":
        trainer = BYOLTrainer(data, args.learning_rate, args.epochs, augmentation_strategy, augmentation_name, args.seed)
    elif args.model == "moco":
        trainer = MoCoTrainer(data, args.learning_rate, args.epochs, augmentation_strategy, augmentation_name, args.seed)
    else:
        raise ValueError("Unsupported model type")
    trainer.train()
    # Save the trained model
    trainer.save_model(model_path)

if __name__ == "__main__":
    main()
