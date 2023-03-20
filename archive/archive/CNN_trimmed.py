import os
import torch
from torch import nn, optim
from torch.utils import data
from torchsummary import summary
from tqdm import tqdm
import cv2
import numpy

from utils.load_data import load_trimmed_labelled
from utils.split_data import split_data
from utils.normalize import normalize

class TrimmedLabelled(data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, i):
        return self.x[i], self.y[i]

class Model(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels = in_channels,
                out_channels = 32,
                kernel_size = 3,
                padding = 1
            ),
            nn.ReLU(),
            
            nn.MaxPool2d(kernel_size = 2),
            
            nn.Conv2d(
                in_channels = 32,
                out_channels = 16,
                kernel_size = 3,
                padding = 1
            ),
            nn.ReLU(),
            
            nn.MaxPool2d(kernel_size = 2),
            
            nn.Conv2d(
                in_channels = 16,
                out_channels = 8,
                kernel_size = 3,
                padding = 1
            ),
            nn.ReLU(),
            
            nn.MaxPool2d(kernel_size = 2),
            
            nn.Conv2d(
                in_channels = 8,
                out_channels = 4,
                kernel_size = 3,
                padding = 1
            ),
            nn.ReLU(),
            
            nn.MaxPool2d(kernel_size = 2),
            
            nn.Conv2d(
                in_channels = 4,
                kernel_size = 2,
                kernel_size = 3,
                padding = 1
            ),
            nn.ReLU(),
            
            nn.MaxPool2d(kernel_size = 2),
            
            nn.Flatten(),
            nn.Linear(in_features = 2 * 4 * 4, out_features = 1),
            nn.Sigmoid(),
        )
        
    def forward(self, input):
        return self.model(input)
    
class Trainer:
    def __init__(
        self,
        model,
        device
    ):
        self.model  = model.to(device)
        self.device = device
        
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters())
        
    def step(
        self,
        train_loader
    ):
        self.model.train()
        
        loss_sum = 0
        for x, y in tqdm(train_loader, desc='Training'):
            x = x.to(self.device)
            y = y.to(self.device)
            
            prediction = self.model(x)
            loss       = self.criterion(prediction, y)
            
            loss_sum += loss.item()
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        return loss_sum / len(train_loader)
    
    def test(
        self,
        test_loader
    ):
        self.model.eval()
        
        loss_sum = 0
        with torch.no_grad():
            for x, y in tqdm(test_loader, desc='Testing'):
                x = x.to(self.device)
                y = y.to(self.device)
                
                prediction = self.model(x)
                loss       = self.criterion(prediction, y)
                
                loss_sum += loss.item()
            
        return loss_sum / len(test_loader)
    
    def fit(
        self,
        train_loader,
        test_loader,
        max_epochs = 300,
        terminate_threshold = 5
    ):  
        min_loss        = float('inf')
        terminate_count = 0
        for epoch in tqdm(range(1, max_epochs + 1), desc='Fitting'):
            train_loss = self.step(train_loader)
            test_loss  = self.test(test_loader)
            
            terminate_count += 1
            if test_loss < min_loss:
                min_loss        = test_loss
                terminate_count = 0
            elif terminate_count > terminate_threshold:
                break
            
            tqdm.write(
                f'Epoch {epoch:>3}: loss={train_loss:>.7f}, ' \
                f'test_loss={test_loss:>.7f}, ' \
                f'min_loss={min_loss:>.7f}'
            )
            
def train():
    model_path  = f'model/CNN_trimmed.pt'
    device      = 'mps' if torch.backends.mps.is_available() else 'cpu'
    train_ratio = 0.8
    batch_size  = 16
    
    train_x, train_y = load_trimmed_labelled()
    train_x = train_x.transpose(0, 3, 1, 2)
    train_x, test_x = split_data(train_x, train_ratio)
    train_y, test_y = split_data(train_y, train_ratio)
    
    print(f'Train data: {train_x.shape} {train_y.shape}')
    print(f'Test data: {test_x.shape} {test_y.shape}')
    
    model = Model(train_x.shape[1])
    summary(model, train_x[0].shape)
    
    train_loader = data.DataLoader(
        TrimmedLabelled(train_x, train_y),
        batch_size = batch_size,
        shuffle = True
    )
    test_loader = data.DataLoader(
        TrimmedLabelled(test_x, test_y),
        batch_size = batch_size
    )
    
    trainer = Trainer(model, device)
    trainer.fit(train_loader, test_loader)
    
    torch.save(trainer.model.to('cpu'), model_path)

if __name__ == '__main__':
    train()
    