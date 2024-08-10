import os
import time
import random
from datetime import timedelta
import torch
import numpy as np
from torch import nn
import torchvision.models as models
from lightly.loss import NTXentLoss
from lightly.models.modules import SimCLRProjectionHead
from utils.utils import plot_training_curve, set_seed

class SimCLR(nn.Module):
    """
    SimCLR model with a ResNet backbone.
    input: backbone(Encoder)
    output: z(Embedding)
    """
    
    def __init__(self, backbone):
        super().__init__()
        
        self.backbone = backbone
        self.projection_head = SimCLRProjectionHead(512, 512, 128)
        
    def forward(self, x):
        
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z



class SimCLRTrainer():
    """
    SimCLR Trainer class
    input: data, learning_rate, epochs, augmentation_strategy, augmentation_name, seed
    output: losses
    """
    # Initialize the SimCLRTrainer class
    def __init__(self, data, learning_rate, epochs, augmentation_strategy, augmentation_name, seed):
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.augmentation_strategy = augmentation_strategy
        self.augmentation_name = augmentation_name
        self.data = data
        self.resnet = models.resnet18()
        self.seed = seed
        set_seed(seed)
        self.backbone = nn.Sequential(*list(self.resnet.children())[:-1])
        self.model = SimCLR(self.backbone)
        self.model.to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.criterion = NTXentLoss()
        self.losses = []

    # training_step() function is used to perform a single training step
    def training_step(self, batch):
        
        total_loss = 0  
        x0 = batch[0].to(self.device)
        x1 = batch[0].to(self.device)
        x1_pos = torch.stack([self.augmentation_strategy(image) for image in x1])
        x1_neg = batch[0].to(self.device)
        random.shuffle(x1_neg)
        z0 = self.model(x0)
        z1_pos = self.model(x1_pos)
        z1_neg = self.model(x1_neg)
        loss_pos = self.criterion(z0, z1_pos)
        loss_neg = -self.criterion(z0, z1_neg)
        loss = loss_pos + loss_neg
        total_loss += loss.item()
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item()

    # train() function is used to train the model
    def train(self):

        start_time = time.time()
        print("Started Training")
        for epoch in range(1, self.epochs + 1):
            total_loss = 0
            for batch in self.data:
                loss = self.training_step(batch)
                total_loss += loss
            avg_loss = total_loss / len(self.data)
            print(f"Epoch [{epoch}/{self.epochs}], Average Loss: {avg_loss:.4f}")
            self.losses.append(avg_loss)
        print("Training finished")
        end_time = time.time()
        elapsed_time_seconds = end_time - start_time
        elapsed_time_formatted = str(timedelta(seconds=elapsed_time_seconds))
        print(f"Training completed in {elapsed_time_formatted}")
        # plot_training_curve(range(1, epoch + 1), self.losses, self.augmentation_name, "/home/jovyan/plots/simclr")
        return self.losses

    # save_model() function is used to save the model
    def save_model(self, model_path):
        
        model_name = f"simclr_model_{self.augmentation_name}.pth"
        model_path = os.path.join(model_path, model_name)
        torch.save(self.model.state_dict(), model_path)
    
    