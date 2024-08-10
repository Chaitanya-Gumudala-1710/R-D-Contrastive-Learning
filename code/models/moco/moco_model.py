import os
import copy
import torch
import time
import random
from datetime import timedelta
import numpy as np
import torchvision.models as models
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch import nn
import lightly.data as data
from lightly.loss import NTXentLoss
from lightly.models.modules import MoCoProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.transforms.moco_transform import MoCoV2Transform
from lightly.utils.scheduler import cosine_schedule
from utils.utils import plot_training_curve, set_seed


class MoCo(nn.Module):
    """
    MoCo model with a ResNet backbone.
    input: backbone(Encoder)
    output: query in forward pass and key in forward_momentum pass
    """
    def __init__(self, backbone):
        super().__init__()

        self.backbone = backbone
        self.projection_head = MoCoProjectionHead(512, 512, 128)

        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

    def forward(self, x):
        query = self.backbone(x).flatten(start_dim=1)
        query = self.projection_head(query)
        return query

    def forward_momentum(self, x):
        key = self.backbone_momentum(x).flatten(start_dim=1)
        key = self.projection_head_momentum(key).detach()
        return key




class MoCoTrainer():
    """
    MoCo Trainer class
    input: data, learning_rate, epochs, augmentation_strategy, augmentation_name, seed
    output: losses
    """
    # Initialize the MoCoTrainer class
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
        self.model = MoCo(self.backbone)
        self.model.to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001)
        self.criterion = NTXentLoss(memory_bank_size=(4096, 128))
        self.losses = []

    # training_step() function is used to perform a single training step
    def training_step(self, batch, momentum_val):
        total_loss = 0
        x_query = batch[0]
        x_key = batch[0]
        update_momentum(self.model.backbone, self.model.backbone_momentum, m=momentum_val)
        update_momentum(self.model.projection_head, self.model.projection_head_momentum, m=momentum_val)
        x_query = x_query.to(self.device)
        x_key = x_key.to(self.device)
        x_key = torch.stack([self.augmentation_strategy(image) for image in x_key])
        x_neg_key = batch[0].to(self.device)
        random.shuffle(x_neg_key)
        query = self.model(x_query)
        key = self.model.forward_momentum(x_key)
        neg_key = self.model.forward_momentum(x_neg_key)
        loss_pos = self.criterion(query, key)
        loss_neg = -self.criterion(query, neg_key)
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
        # plot_training_curve(range(1, epoch + 1), self.losses, self.augmentation_name, "/home/jovyan/plots/moco")
        return self.losses
    
    # save_model() function is used to save the model    
    def save_model(self, model_path):
        model_name = f"moco_model_{self.augmentation_name}.pth"
        model_path = os.path.join(model_path, model_name)
        torch.save(self.model.state_dict(), model_path)