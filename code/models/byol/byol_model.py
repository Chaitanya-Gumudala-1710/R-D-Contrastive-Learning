import os
import copy
import time
from datetime import timedelta
import torch
import random
import numpy as np
from torch import nn
import torchvision.models as models
from lightly.loss import NegativeCosineSimilarity
from lightly.models.modules import BYOLPredictionHead, BYOLProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.utils.scheduler import cosine_schedule
from utils.utils import plot_training_curve, set_seed

class BYOL(nn.Module):
    """
    BYOL model with a ResNet backbone.
    input: backbone(Encoder)
    output: p(Prediction) in forward pass and z(Embedding) in forward_momentum pass
    """
    
    def __init__(self, backbone):
        super().__init__()

        self.backbone = backbone
        self.projection_head = BYOLProjectionHead(512, 1024, 128)
        self.prediction_head = BYOLPredictionHead(128, 1024, 128)

        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

    def forward(self, x):
        
        y = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(y)
        p = self.prediction_head(z)
        return p

    def forward_momentum(self, x):
                        
        y = self.backbone_momentum(x).flatten(start_dim=1)
        z = self.projection_head_momentum(y)
        z = z.detach()
        return z


class BYOLTrainer():
    """
    BYOL Trainer class
    input: data, learning_rate, epochs, augmentation_strategy, augmentation_name, seed
    output: losses
    """
    # Initialize the BYOLTrainer class
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
        self.model = BYOL(self.backbone)
        self.model.to(self.device)
        self.criterion = NegativeCosineSimilarity()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001)
        self.losses = []

    # training_step() function is used to perform a single training step
    def training_step(self, batch, momentum_val):
        total_loss = 0
        x0 = batch[0].to(self.device)
        x1 = torch.stack([self.augmentation_strategy(image) for image in x0]).to(self.device)
        
        update_momentum(self.model.backbone, self.model.backbone_momentum, m=momentum_val)
        update_momentum(self.model.projection_head, self.model.projection_head_momentum, m=momentum_val)
        
        p0 = self.model(x0)
        z0 = self.model.forward_momentum(x0)
        p1 = self.model(x1)
        z1 = self.model.forward_momentum(x1)
        
        loss = 0.5 * (self.criterion(p0, z1) + self.criterion(p1, z0))
        
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
            momentum_val = cosine_schedule(epoch, self.epochs, 0.996, 1)
            total_loss = 0
            for batch in self.data:
                loss = self.training_step(batch, momentum_val)
                total_loss += loss
            avg_loss = total_loss / len(self.data)
            print(f"Epoch [{epoch}/{self.epochs}], Average Loss: {avg_loss:.4f}")
            self.losses.append(avg_loss)
        print("Training finished")
        end_time = time.time()
        elapsed_time_seconds = end_time - start_time
        elapsed_time_formatted = str(timedelta(seconds=elapsed_time_seconds))
        print(f"Training completed in {elapsed_time_formatted}")
        # plot_training_curve(range(1, epoch + 1), self.losses, self.augmentation_name, "/home/jovyan/plots/byol")
        return self.losses
    
    # save_model() function is used to save the model
    def save_model(self, model_path):
        model_name = f"byol_model_{self.augmentation_name}.pth"
        model_path = os.path.join(model_path, model_name)
        torch.save(self.model.state_dict(), model_path)
