import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import GaussianBlur
from torchsummary import summary
from tqdm import tqdm
import os
import numpy
import cv2
import random

from utils.split_data import split_data
from utils.get_reflectance import get_reflectance
from utils.normalize import normalize

DATA_PATH = 'data/trimmed'
IMAGES = [
    '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'
]

torch.autograd.set_detect_anomaly(True)

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.MaxPool2d(kernel_size = 2),
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size = 3, padding = 1,
                padding_mode = 'replicate'
            ),
            nn.BatchNorm2d(out_channels),
            nn.ELU()
        )
        
    def forward(self, x):
        return self.model(x)
    
class UpBlock(nn.Module):
    def __init__(self, in_channels, add_channels, out_channels):
        super().__init__()
        
        self.up = nn.Sequential(
            nn.Upsample(scale_factor = 2),
            nn.Conv2d(
                in_channels, in_channels // 2,
                kernel_size = 3, padding = 1
            ),
            nn.BatchNorm2d(in_channels // 2),
            nn.ELU()
        )
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels // 2 + add_channels, out_channels,
                kernel_size = 3, padding = 1
            ),
            nn.BatchNorm2d(out_channels),
            nn.ELU()
        )
        
    def forward(self, x1, x2):
        x1 = self.up(x1)
        return self.conv(torch.cat((x1, x2), dim = 1))

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.entrance = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels = 24,
                kernel_size = 5, padding = 2,
                padding_mode = 'replicate'
            ),
            nn.BatchNorm2d(24),
            nn.ELU()
        )
        self.down1 = DownBlock(in_channels = 24, out_channels = 32)
        self.down2 = DownBlock(in_channels = 32, out_channels = 48)
        self.down3 = DownBlock(in_channels = 48, out_channels = 64)
        self.down4 = DownBlock(in_channels = 64, out_channels = 96)
        
        self.up1  = UpBlock(in_channels = 96, add_channels = 64, out_channels = 64)
        self.up2  = UpBlock(in_channels = 64, add_channels = 48, out_channels = 48)
        self.up3  = UpBlock(in_channels = 48, add_channels = 32, out_channels = 32)
        self.up4  = UpBlock(in_channels = 32, add_channels = 24, out_channels = 24)
        self.exit = nn.Sequential(
            nn.Conv2d(
                in_channels = 24, out_channels = out_channels,
                kernel_size = 3, padding = 1
            ),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x1 = self.entrance(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        y1 = self.up1(x5, x4)
        y2 = self.up2(y1, x3)
        y3 = self.up3(y2, x2)
        y4 = self.up4(y3, x1)
        y  = self.exit(y4)
        
        return y
    
class Autoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        
        self.encoder:UNet = encoder
        self.decoder:UNet = decoder
        self.blurer       = GaussianBlur(kernel_size = 21, sigma = 10)
        
    def forward(self, x):
        blurred = self.blurer(x)
        encoded = self.encoder(x)
        x = torch.multiply(blurred, encoded)
        return self.decoder(x)
    
class Images(Dataset):
    def __init__(self, files):
        self.files  = files
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, i):
        with numpy.load(os.path.join(DATA_PATH, self.files[i])) as f:
            images1 = torch.tensor(f['x'], dtype = torch.float) / 255
        
        images2 = images1[numpy.random.permutation(images1.shape[0])]
        return images1, images2
    
class Decoy:
    def __enter__(self):
        pass
    
    def __exit__(self, _, __, ___):
        pass
    
class Trainer:
    def __init__(self, autoencoder, train_images, valid_images, batch_size, device):
        self.autoencoder:Autoencoder = autoencoder.to(device, non_blocking = True)
        
        self.train_loader = DataLoader(
            Images(train_images),
            batch_size,
            shuffle = True,
            num_workers = 4
        )
        self.valid_loader = DataLoader(
            Images(valid_images),
            batch_size,
            shuffle = True,
            num_workers = 4
        )
        
        self.encoder_criterion = nn.MSELoss()
        self.model_criterion   = nn.MSELoss()
        
        self.optimizer = optim.Adam(self.autoencoder.parameters())
        
        self.device = device
        
    def step_encoder(self, x1, x2):
        x1_features = self.autoencoder.encoder(x1)
        x2_features = self.autoencoder.encoder(x2)
        
        encoder_loss = self.encoder_criterion(x1_features, x2_features)
        return encoder_loss
    
    def step_model(self, x):
        y = self.autoencoder(x)
        
        model_loss = self.model_criterion(y, x)
        return model_loss
        
    def step(self, train = True, epoch = 0):
        if train:
            self.autoencoder.train()
            no_grad = Decoy
        else:
            self.autoencoder.eval()
            no_grad = torch.no_grad
        
        encoder_loss_sum = 0
        model_loss_sum   = 0
        with no_grad():
            for i, (x1, x2) in tqdm(
                list(enumerate(self.train_loader if train else self.valid_loader)),
                desc = 'Training' if train else 'Validating'
            ):
                x1 = x1.to(self.device)
                x2 = x2.to(self.device)
                
                model_loss:torch.Tensor = self.step_model(x1)
                model_loss_sum += model_loss.item()
                if train:
                    self.optimizer.zero_grad()
                    model_loss.backward()
                    self.optimizer.step()
                
                '''
                encoder_loss:torch.Tensor = self.step_encoder(x1, x2)
                encoder_loss_sum += encoder_loss.item()
                if train:
                    self.optimizer.zero_grad()
                    encoder_loss.backward()
                    self.optimizer.step()
                '''
                
                if not train:
                    in_image  = x1[0][0].cpu().detach().numpy() * 255
                    blurred   = self.autoencoder.blurer(x1)[0][0].cpu().detach().numpy() * 255
                    features  = self.autoencoder.encoder(x1)[0][0].cpu().detach().numpy() * 255
                    out_image = self.autoencoder(x1)[0][0].cpu().detach().numpy() * 255
                    
                    cv2.imwrite(f'rollout/unsupervised/{epoch}_{i}_in.png', in_image)
                    cv2.imwrite(f'rollout/unsupervised/{epoch}_{i}_blurred.png', blurred)
                    cv2.imwrite(f'rollout/unsupervised/{epoch}_{i}_features.png', features)
                    cv2.imwrite(f'rollout/unsupervised/{epoch}_{i}_out.png', out_image)
            
        data_count = len(self.train_loader if train else self.valid_loader)
        return \
            encoder_loss_sum / data_count, \
            model_loss_sum / data_count
    
    def run(self, max_epochs = 100):
        for epoch in tqdm(range(1, max_epochs + 1), desc = 'Epochs'):
            train_encoder_loss, train_autoencoder_loss = self.step()
            valid_encoder_loss, valid_autoencoder_loss = self.step(train = False, epoch = epoch)
            
            tqdm.write(
                f'Epoch {epoch:>3}: ' \
                f'train_loss = ({train_encoder_loss:>.7f}, {train_autoencoder_loss:>.7f}), ' \
                f'valid_loss = ({valid_encoder_loss:>.7f}, {valid_autoencoder_loss:>.7f})'
            )
    
def train():
    batch_size = 256
    device     = 'cpu'
    
    encoder     = UNet(in_channels = 21, out_channels = 1)
    decoder     = UNet(in_channels = 21, out_channels = 21)
    autoencoder = Autoencoder(encoder, decoder)
    
    summary(autoencoder, (21, 128, 128))
    
    data = os.listdir(DATA_PATH)
    data = [datum for datum in data if '.npz' in datum]
    random.shuffle(data)
    data = data[:10000]
    train_images, valid_images = split_data(data)
    
    trainer = Trainer(autoencoder, train_images, valid_images, batch_size, device)
    trainer.run(max_epochs = 100)
    
    torch.save(trainer.autoencoder.encoder, 'model/unsupervised_encoder.pt')
    torch.save(trainer.autoencoder.decoder, 'model/unsupervised_decoder.pt')
    
def rollout(file, ae):
    reflectance = get_reflectance(os.path.join('data', file)).transpose(2, 0, 1)
    image = []
    
    _, height, width = reflectance.shape
    for h in tqdm(range(0, height - 128, 128)):
        row = []
        for w in range(0, width - 128, 128):
            trimmed = reflectance[:, h:h + 128, w:w + 128]
            for c, channel in enumerate(trimmed):
                trimmed[c] = normalize(channel)
            row.append(trimmed)
        
        row = numpy.array(row, dtype = numpy.float32)
        
        # prediction = ae.encoder(torch.tensor(row)).detach().numpy().squeeze()
        # prediction = numpy.concatenate(prediction, axis = 1)
        # image.append(prediction)
        
        regen = ae(torch.tensor(row)).detach().numpy()[:, 0]
        regen = numpy.concatenate(regen, axis = 1)
        image.append(regen)
        
    # image = normalize(numpy.concatenate(image)) * 255
    # cv2.imwrite(f'rollout/unsupervised/{file}.png', image)
    
    image = normalize(numpy.concatenate(image)) * 255
    cv2.imwrite(f'rollout/unsupervised/{file}_regen.png', image)
    
def infer():
    encoder = torch.load('model/unsupervised_encoder.pt', map_location = 'cpu')
    decoder = torch.load('model/unsupervised_decoder.pt', map_location = 'cpu')
    ae      = Autoencoder(encoder, decoder)
    summary(encoder, (21, 128, 128))
    summary(ae, (21, 128, 128))
    
    for image in IMAGES:
        rollout(f'{image}', ae)
    
if __name__ == '__main__':
    infer()