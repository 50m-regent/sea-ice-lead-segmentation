import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchsummary import summary
from tqdm import tqdm
import os
import random
import numpy
import cv2

from utils.split_data import split_data

DATA_PATH = 'data/trimmed'

class Generator(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels = 32,
                kernel_size = 7,
                padding = 3,
                padding_mode = 'reflect'
            ),
            nn.InstanceNorm2d(32),
            nn.ReLU(),
            
            nn.Conv2d(
                in_channels = 32,
                out_channels = 64,
                kernel_size = 3,
                stride = 2,
                padding = 1
            ),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(
                in_channels = 64,
                out_channels = 64,
                kernel_size = 3,
                stride = 2,
                padding = 1
            ),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            
            nn.Upsample(size = 64),
            
            nn.Conv2d(
                in_channels = 64,
                out_channels = 32,
                kernel_size = 3,
                padding = 1
            ),
            nn.InstanceNorm2d(32),
            nn.ReLU(),
            
            nn.Upsample(size = 128),
            
            nn.Conv2d(
                in_channels = 32,
                out_channels = 16,
                kernel_size = 3,
                padding = 1
            ),
            nn.InstanceNorm2d(16),
            nn.ReLU(),
            
            nn.Conv2d(
                in_channels = 16,
                out_channels = 1,
                kernel_size = 3,
                padding = 1
            ),
            nn.Sigmoid(),
        )
        
    def mask(self, x):
        return torch.where(self.model(x) > 0.5, 1, 0)
        
    def forward(self, x):
        return torch.multiply(x, self.mask(x))
    
class Discriminator(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels = 32,
                kernel_size = 5,
                stride = 2
            ),
            nn.MaxPool2d(kernel_size = 2),
            nn.ReLU(),
            
            nn.Conv2d(
                in_channels = 32,
                out_channels = 64,
                kernel_size = 5,
                stride = 2
            ),
            nn.MaxPool2d(kernel_size = 2),
            nn.ReLU(),
            
            nn.Conv2d(
                in_channels = 64,
                out_channels = 128,
                kernel_size = 5,
                stride = 2
            ),
            nn.MaxPool2d(kernel_size = 2),
            nn.ReLU(),
            
            nn.Flatten(),
            nn.Linear(in_features = 128, out_features = 1),
            nn.Sigmoid(),
        )
        
    def forward(self, x):
        return self.model(x)

class Images(Dataset):
    def __init__(self, x):
        self.x = x
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, i):
        return self.x[i]

class GANTrainer:
    def __init__(
        self,
        generator:Generator,
        discriminator:Discriminator,
        train_images,
        valid_images,
        batch_size = 128,
        device = 'cpu'
    ):
        self.generator:Generator         = generator.to(device)
        self.discriminator:Discriminator = discriminator.to(device)
        
        self.train_loader = DataLoader(
            Images(train_images),
            batch_size,
            shuffle = True,
            drop_last = True
        )
        self.valid_loader = DataLoader(
            Images(valid_images),
            batch_size,
            drop_last = True
        )
        self.label_0 = torch.zeros((batch_size, 1), device = device)
        self.label_1 = torch.ones((batch_size, 1), device = device)
        
        self.criterion = nn.BCELoss().to(device)
        
        self.generator_optimizer     = optim.Adam(self.generator.parameters())
        self.discriminator_optimizer = optim.Adam(self.discriminator.parameters())
        
        self.device = device
        
    def train_generator(self, x):
        fake_x = self.generator(x)
        fake_y = self.discriminator(fake_x)
        
        generator_loss:torch.Tensor = self.criterion(
            fake_y, self.label_0
        )
        
        self.generator_optimizer.zero_grad()
        generator_loss.backward()
        self.generator_optimizer.step()
        
        return generator_loss.item()
    
    def train_discriminator(self, x):
        fake_x = self.generator(x)
        true_y = self.discriminator(x)
        fake_y = self.discriminator(fake_x)
        
        true_loss:torch.Tensor = self.criterion(
            true_y, self.label_0
        )
        fake_loss:torch.Tensor = self.criterion(
            fake_y, self.label_1
        )
        
        self.discriminator_optimizer.zero_grad()
        true_loss.backward()
        fake_loss.backward()
        self.discriminator_optimizer.step()
        
        # return true_loss.item(), fake_loss.item()
        return true_loss.item() + fake_loss.item()
    
    def train_step(self):
        self.generator.train()
        self.discriminator.train()
        
        generator_loss_sum     = 0
        discriminator_loss_sum = 0
        for x in tqdm(self.train_loader, desc = 'Training'):
            x = x.to(self.device)
            
            generator_loss_sum     += self.train_generator(x)
            discriminator_loss_sum += self.train_discriminator(x)
            
        return generator_loss_sum / len(self.train_loader), discriminator_loss_sum / len(self.train_loader)
    
    def valid_step(self, epoch):
        self.generator.eval()
        self.discriminator.eval()
        
        generator_loss_sum = 0
        with torch.no_grad():
            for i, x in tqdm(list(enumerate(self.valid_loader)), desc = 'Validating'):
                x = x.to(self.device)
                
                fake_x = self.generator(x)
                fake_y = self.discriminator(fake_x)
                
                generator_loss:torch.Tensor = self.criterion(
                    fake_y, torch.zeros(fake_y.shape, device = self.device)
                )
                generator_loss_sum += generator_loss.item()
                
                cv2.imwrite(f'gan/{epoch}_{i}_true.png', x[0][0].cpu().detach().numpy())
                cv2.imwrite(f'gan/{epoch}_{i}_fake.png', fake_x[0][0].cpu().detach().numpy())
                cv2.imwrite(f'gan/{epoch}_{i}_mask.png', self.generator.mask(x)[0][0].cpu().detach().numpy() * 255)
            
        return generator_loss_sum / len(self.valid_loader)
    
    def run(self, max_epochs = 100):
        for epoch in tqdm(range(1, max_epochs + 1), desc = 'Epochs'):
            generator_train_loss, discriminator_train_loss = self.train_step()
            generator_valid_loss = self.valid_step(epoch)
            
            tqdm.write(
                f'Epoch {epoch:>3}: ' \
                f'train_loss = ({generator_train_loss:>.7f}, {discriminator_train_loss:>.7f}), ' \
                f'valid_loss = {generator_valid_loss:>.7f}'
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
            
    return numpy.asarray(images, dtype = numpy.float32)
    
def train():
    images = load_images(n = 10000)
    train_images, valid_images = split_data(images)
    print(f'Train data: {train_images.shape}')
    print(f'Valid data: {valid_images.shape}')
    
    generator     = Generator(images.shape[1])
    print('Generator')
    summary(generator, images.shape[1:])
    
    discriminator = Discriminator(images.shape[1])
    print('Discriminator')
    summary(discriminator, images.shape[1:])
    
    trainer = GANTrainer(generator, discriminator, train_images, valid_images, device = 'mps')
    trainer.run()
    
if __name__ == '__main__':
    train()