import os
import torch
from torch import nn, optim
from torch.utils import data
from torchsummary import summary
from tqdm import tqdm
import cv2
import numpy
from sklearn.svm import OneClassSVM

from utils.load_data import load_trimmed
from utils.split_data import split_data
from utils.normalize import normalize

class Trimmed(data.Dataset):
    def __init__(self, x):
        self.x = x

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, i):
        return self.x[i]

class Encoder(nn.Module):
    def __init__(self, in_channels, features):
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
            nn.BatchNorm2d(32),
            
            nn.Conv2d(
                in_channels = 32,
                out_channels = 16,
                kernel_size = 3,
                padding = 1
            ),
            nn.ReLU(),
            
            nn.MaxPool2d(kernel_size = 2),
            nn.BatchNorm2d(16),
            
            nn.Conv2d(
                in_channels = 16,
                out_channels = 8,
                kernel_size = 3,
                padding = 1
            ),
            nn.ReLU(),
            
            nn.MaxPool2d(kernel_size = 2),
            nn.BatchNorm2d(8),
            
            nn.Conv2d(
                in_channels = 8,
                out_channels = 4,
                kernel_size = 3,
                padding = 1
            ),
            nn.ReLU(),
            
            nn.MaxPool2d(kernel_size = 2),
            nn.BatchNorm2d(4),
            
            nn.Flatten(),
            nn.Linear(in_features = 4 * 8 * 8, out_features = features),
        )
        
    def forward(self, input):
        return self.model(input)
    
class Decoder(nn.Module):
    def __init__(self, out_channels, features):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(in_features = features, out_features = 4 * 8 * 8),
            nn.ReLU(),
            nn.Unflatten(dim = 1, unflattened_size=(4, 8, 8)),
            
            nn.ConvTranspose2d(
                in_channels = 4,
                out_channels = 4,
                kernel_size = 5,
                stride = 2,
                padding = 2
            ),
            nn.ReLU(),
            
            nn.ConvTranspose2d(
                in_channels = 4,
                out_channels = 8,
                kernel_size = 5,
                stride = 2,
                padding = 1
            ),
            nn.ReLU(),
            
            nn.ConvTranspose2d(
                in_channels = 8,
                out_channels = 16,
                kernel_size = 5,
                stride = 2,
                padding = 1
            ),
            nn.ReLU(),
            
            nn.ConvTranspose2d(
                in_channels = 16,
                out_channels = 32,
                kernel_size = 4,
                stride = 2,
                padding = 1
            ),
            nn.ReLU(),
            
            nn.ConvTranspose2d(
                in_channels = 32,
                out_channels = out_channels,
                kernel_size = 3
            ),
            nn.ReLU(),
        )
        
    def forward(self, input):
        return self.model(input)

class AutoEncoder(nn.Module):
    def __init__(
        self,
        encoder,
        decoder
    ):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, input):
        return self.decoder(self.encoder(input))
    
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
        for x in tqdm(train_loader, desc='Training'):
            x = x.to(self.device)
            
            prediction = self.model(x)
            loss       = self.criterion(prediction, x)
            
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
            for x in tqdm(test_loader, desc='Testing'):
                x = x.to(self.device)
                
                prediction = self.model(x)
                loss       = self.criterion(prediction, x)
                
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
            
def train_auto_encoder(features):
    model_path   = f'model/AE_trimmed_f{features}.pt'
    encoder_path = f'model/AE_trimmed_f{features}_encoder.pt'
    device       = 'mps' if torch.backends.mps.is_available() else 'cpu'
    train_ratio  = 0.8
    batch_size   = 64
    
    train_x = load_trimmed()
    train_x = train_x.transpose(0, 3, 1, 2)
    train_x, test_x = split_data(train_x, train_ratio)
    
    print(f'Train data: {train_x.shape}')
    print(f'Test data: {test_x.shape}')
    
    model = AutoEncoder(
        Encoder(train_x.shape[1], features),
        Decoder(train_x.shape[1], features)
    )
    summary(model, train_x[0].shape)
    
    train_loader = data.DataLoader(
        Trimmed(train_x),
        batch_size=batch_size,
        shuffle=True
    )
    test_loader = data.DataLoader(
        Trimmed(test_x),
        batch_size=batch_size
    )
    
    trainer = Trainer(model, device)
    trainer.fit(train_loader, test_loader)
    
    torch.save(trainer.model.to('cpu'), model_path)
    torch.save(trainer.model.encoder.to('cpu'), encoder_path)
    
def inference(features, n = 100):
    model_path = f'model/AE_trimmed_f{features}.pt'
    assert os.path.exists(model_path)
    
    model = torch.load(model_path)
    data = load_trimmed(n = n)
    
    for i, before in tqdm(list(enumerate(data))):
        x = before.transpose(2, 0, 1)
        y = model(torch.tensor(numpy.array([x]))).detach().numpy()
        y = y[0].transpose(1, 2, 0)
        
        cv2.imwrite(
            f'AE_trimmed/f{features}/{i:>03}_before.png',
            normalize(before[:, :, 0]) * 255
        )
        cv2.imwrite(
            f'AE_trimmed/f{features}/{i:>03}_after.png',
            normalize(y[:, :, 0]) * 255
        )
        
def classify(features):
    model_path = f'model/AE_trimmed_f{features}_encoder.pt'
    assert os.path.exists(model_path)
    encoder = torch.load(model_path)
    
    train_x = load_trimmed(n = 1000)
    encoded = encoder(
        torch.tensor(train_x.transpose(0, 3, 1, 2))
    ).detach().numpy()
    
    svm = OneClassSVM(verbose = True)
    svm.fit(encoded)
    predictions = svm.predict(encoded)
    
    i0, i1 = 0, 0
    for image, prediction in tqdm(list(zip(train_x, predictions))):
        if 1 == prediction:
            cv2.imwrite(
                f'AE_trimmed/1/{i1:>04}.png',
                normalize(image[:, :, 0]) * 255
            )
            
            i1 += 1
        else:
            cv2.imwrite(
                f'AE_trimmed/0/{i0:>04}.png',
                normalize(image[:, :, 0]) * 255
            )
            
            i0 += 1

if __name__ == '__main__':
    # classify(features = 16)
    
    # for features in [4, 8, 16, 32]:
    #     train_auto_encoder(features)
    #     inference(features)
    
    train_auto_encoder(features = 8)
    