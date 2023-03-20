import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import GaussianBlur
from torchsummary import summary
from tqdm import tqdm
import os
import numpy
import random
import cv2

from utils.split_data import split_data

DATA_PATH = 'data/trimmed'

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.MaxPool2d(kernel_size = 2),
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size = 3, padding = 1,
                padding_mode = 'replicate'
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        
    def forward(self, x):
        return self.conv(x)
    
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.up = nn.Sequential(
            nn.Upsample(scale_factor = 2),
            nn.ConvTranspose2d(
                in_channels, out_channels,
                kernel_size = 3, padding = 1
            ),
            nn.ReLU()
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels,
                kernel_size = 3, padding = 1
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        
    def forward(self, x1, x2):
        x1 = self.up(x1)
        return self.deconv(torch.cat((x1, x2), dim = 1))

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.entrance = nn.Sequential(
            nn.Conv2d(
                in_channels = in_channels, out_channels = 32,
                kernel_size = 3, padding = 1,
                padding_mode = 'replicate'
            ),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.down1 = DownBlock(in_channels = 32,  out_channels = 64)
        self.down2 = DownBlock(in_channels = 64,  out_channels = 128)

        self.up1  = UpBlock(in_channels = 128,  out_channels = 64)
        self.up2  = UpBlock(in_channels = 64,   out_channels = 32)
        self.exit = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels = 32, out_channels = out_channels,
                kernel_size = 3, padding = 1
            ),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x1 = self.entrance(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        
        y1 = self.up1(x3, x2)
        y2 = self.up2(y1, x1)
        y  = self.exit(y2)
        
        return y
    
class Autoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        
        self.encoder:UNet = encoder
        self.decoder:UNet = decoder
        self.blurer       = GaussianBlur(kernel_size = 21, sigma = 10)
        
    def forward(self, x):
        blurred = self.blurer(x)
        return self.decoder(torch.multiply(blurred, self.encoder(x)))
    
class Images(Dataset):
    def __init__(self, x):
        self.x = x
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, i):
        return torch.tensor(self.x[i], dtype = torch.float) / 255
    
class Decoy:
    def __enter__(self):
        pass
    
    def __exit__(self, _, __, ___):
        pass
    
class Trainer:
    def __init__(self, autoencoder, train_images, valid_images, batch_size, device, lr = 0.001):
        self.device = device
        
        self.autoencoder:Autoencoder = autoencoder.to(device)
        
        self.train_loader = DataLoader(
            Images(train_images),
            batch_size,
            shuffle = True
        )
        self.valid_loader = DataLoader(
            Images(valid_images),
            batch_size,
            shuffle = True
        )
        
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.autoencoder.parameters(), lr)
        
    def step(self, train = True, epoch = 0):
        if train:
            self.autoencoder.train()
            no_grad = Decoy
        else:
            self.autoencoder.eval()
            no_grad = torch.no_grad
        
        loss_sum = 0
        with no_grad():
            for i, x in tqdm(
                list(enumerate(self.train_loader if train else self.valid_loader)),
                desc = 'Training' if train else 'Validating'
            ):
                x = x.to(self.device)
                
                y                 = self.autoencoder(x)
                loss:torch.Tensor = self.criterion(x, y)
                
                loss_sum += loss.item()
                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                
                if not train:
                    in_image  = x[0][0].cpu().detach().numpy() * 255
                    blurred   = self.autoencoder.blurer(x)[0][0].cpu().detach().numpy() * 255
                    features  = self.autoencoder.encoder(x)[0][0].cpu().detach().numpy() * 255
                    out_image = self.autoencoder(x)[0][0].cpu().detach().numpy() * 255
                    
                    cv2.imwrite(f'rollout/unsupervised/{epoch}_{i}_in.png', in_image)
                    cv2.imwrite(f'rollout/unsupervised/{epoch}_{i}_blurred.png', blurred)
                    cv2.imwrite(f'rollout/unsupervised/{epoch}_{i}_features.png', features)
                    cv2.imwrite(f'rollout/unsupervised/{epoch}_{i}_out.png', out_image)
            
        data_count = len(self.train_loader if train else self.valid_loader)
        return loss_sum / data_count
    
    def run(self, max_epochs = 100):
        for epoch in tqdm(range(1, max_epochs + 1), desc = 'Epochs'):
            train_loss = self.step()
            valid_loss = self.step(train = False, epoch = epoch)
            
            tqdm.write(
                f'Epoch {epoch:>3}: ' \
                f'train_loss = {train_loss:>.7f}, ' \
                f'valid_loss = {valid_loss:>.7f}'
            )
    
def load_images(n = -1):
    files = os.listdir(DATA_PATH)
    files = [file for file in files if '.npz' in file]
    random.shuffle(files)
    
    files = files[:len(files) if -1 == n else n]
    
    images = []
    for file in tqdm(files, desc = 'Loading'):
        with numpy.load(os.path.join(DATA_PATH, file)) as f:
            images.append(f['x'])
            
    return numpy.asarray(images, dtype = numpy.uint8)
    
def train():
    batch_size = 256
    device     = 'mps'
    
    encoder     = UNet(in_channels = 21, out_channels = 1)
    decoder     = UNet(in_channels = 21, out_channels = 21)
    autoencoder = Autoencoder(encoder, decoder)
    
    summary(autoencoder, (21, 128, 128))
    
    images = load_images(n = 10000)
    train_images, valid_images = split_data(images)
    
    trainer = Trainer(autoencoder, train_images, valid_images, batch_size, device)
    trainer.run()
    
    torch.save(trainer.autoencoder.encoder, 'model/unsupervised_encoder.pt')
    torch.save(trainer.autoencoder.decoder, 'model/unsupervised_decoder.pt')
    
if __name__ == '__main__':
    train()