import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchsummary import summary
import os
import numpy
from tqdm import tqdm
import cv2
import random

from utils.split_data import split_data

class Patcher(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels = 64,
                kernel_size = 4, stride = 2, padding = 1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(
                in_channels = 64, out_channels = 128,
                kernel_size = 4, stride = 2, padding = 1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
    def forward(self, x):
        return self.model(x)
    
class Classifier(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU()
        )
        
    def forward(self, x):
        return self.model(x)
    
class Model(nn.Module):
    def __init__(self, in_channels, features, classes, device):
        super().__init__()
        
        patcher = Patcher(in_channels).to(device)
        self.representor = nn.Sequential(
            patcher,
            nn.Conv2d(128, features, kernel_size = 1),
            nn.ReLU()
        ).to(device)
        self.body = nn.Sequential(
            patcher,
            nn.Conv2d(
                in_channels = 128, out_channels = 256,
                kernel_size = 4, stride = 2, padding = 1
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.Flatten(),
            nn.Linear(in_features = 256 * 16 * 16, out_features = 1024),
            nn.ReLU(),
            
            nn.Linear(in_features = 1024, out_features = 64),
            nn.ReLU()
        ).to(device)
        self.heads = [
            Classifier(64, features).to(device) for _ in range(classes)
        ]
        self.softmax = nn.Softmax(dim = 2).to(device)
        
    def forward(self, x):
        local_features = self.representor(x).permute((0, 2, 3, 1))
        
        x = self.body(x)
        global_features = torch.stack([head(x) for head in self.heads]).permute((1, 2, 0))
        
        volumes = torch.stack([
            self.softmax(torch.matmul(local_feature, global_feature))
            for local_feature, global_feature in zip(local_features, global_features)
        ])
        
        feature_assignment = torch.stack([
            torch.matmul(volume, global_feature)
            for volume, global_feature in zip(volumes, global_features.transpose(1, 2))
        ])
        
        return local_features, feature_assignment, torch.argmax(volumes, dim = 3)
    
class Images(Dataset):
    def __init__(self, images):
        self.images = []
        
        for image in tqdm(images):
            with numpy.load(image) as f:
                self.images.append(f['x'].astype(numpy.float32) / 255)
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, i):
        return self.images[i]
    
class Decoy:
    def __enter__(self):
        pass
    
    def __exit__(self, _, __, ___):
        pass
    
class Trainer:
    def __init__(self, model, train_loader, valid_loader, device):
        self.model:Model = model
        self.device      = device
        
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters())
        
    def step(self, train = True, epoch = 0):
        if train:
            self.model.train()
            no_grad = Decoy
        else:
            self.model.eval()
            no_grad = torch.no_grad
        
        loss_sum = 0
        with no_grad():
            for i, x in tqdm(
                list(enumerate(self.train_loader if train else self.valid_loader)),
                desc = 'Training' if train else 'Validating'
            ):
                x:torch.Tensor = x.to(self.device)
            
                local_features, feature_assignment, prediction = self.model(x)
                
                loss:torch.Tensor = self.criterion(local_features, feature_assignment)
                loss_sum         += loss.item()
                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                
                if not train:
                    in_image  = x[0][0].cpu().detach().numpy() * 255
                    out_image = prediction[0].cpu().detach().numpy() * 255
                    
                    cv2.imwrite(f'rollout/infoseg/{epoch}_{i}_in.png', in_image)
                    cv2.imwrite(f'rollout/infoseg/{epoch}_{i}_out.png', out_image)
            
        data_count = len(self.train_loader if train else self.valid_loader)
        return loss_sum / data_count
    
    def run(self, max_epochs = 100):
        for epoch in tqdm(range(1, max_epochs + 1), desc = 'Epochs', leave = True):
            train_loss = self.step()
            valid_loss = self.step(train = False, epoch = epoch)
            
            tqdm.write(
                f'Epoch {epoch:>3}: ' \
                f'train_loss = {train_loss:>.7f}, ' \
                f'valid_loss = {valid_loss:>.7f}'
            )
    
def train():
    features   = 32
    classes    = 2
    batch_size = 64
    device     = 'mps'
    
    model = Model(21, features, classes, 'cpu')
    summary(model, (21, 128, 128))
    model = Model(21, features, classes, device)
    
    images = os.listdir('data/trimmed')
    images = [f'data/trimmed/{image}' for image in images if '.npz' in image]
    random.shuffle(images)
    train_images, valid_images = split_data(images[:10000])
    train_loader = DataLoader(
        Images(train_images),
        batch_size,
        shuffle = True
    )
    valid_loader = DataLoader(
        Images(valid_images),
        batch_size,
        shuffle = True
    )
    
    trainer = Trainer(model, train_loader, valid_loader, device)
    trainer.run()
    
if __name__ == '__main__':
    train()