import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import GaussianBlur
from torchsummary import summary
import numpy
from tqdm import tqdm
import cv2
import os

from utils.split_data import split_data
from utils.load_data import load_test_data
from utils.get_reflectance import get_reflectance

IMAGE_PATHS = [
    '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'
]
    
class Encoder(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(in_features, out_features = 64),
            nn.ReLU(),
            
            nn.Linear(in_features = 64, out_features = 128),
            nn.ReLU(),
            
            nn.Linear(in_features = 128, out_features = out_features),
            nn.ReLU()
        )
        
    def forward(self, x):
        return self.model(x)
    
class Decoder(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(in_features, out_features = 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            nn.Linear(in_features = 64, out_features = out_features),
            nn.ReLU(),
        )
        
    def forward(self, x):
        return self.model(x)
    
class Classifier(nn.Module):
    def __init__(self, encoder, features):
        super().__init__()
        
        self.model = nn.Sequential(
            encoder,
            
            nn.Linear(features, out_features = 64),
            nn.ReLU(),
            
            nn.Linear(64, out_features = 2)
        )
        
    def forward(self, x):
        return self.model(x)
    
class Autoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        
        self.model = nn.Sequential(
            encoder,
            decoder
        )
        
    def forward(self, x):
        return self.model(x)
    
class Decoy:
    def __enter__(self):
        pass
    
    def __exit__(self, _, __, ___):
        pass
    
class Pixel(Dataset):
    def __init__(self, x, y = None):
        self.x = x
        self.y = y
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, i):
        if None is self.y:
            return self.x[i]
        else:
            return self.x[i], self.y[i]
    
class Trainer:
    def __init__(self, classifier, autoencoder, unlabelled_data, labelled_data, batch_size, device):
        self.device = device
        
        self.classifier:Classifier   = classifier.to(device)
        self.autoencoder:Autoencoder = autoencoder.to(device)
        
        unlabelled_train_data, unlabelled_valid_data = split_data(unlabelled_data)
        
        labelled_x, labelled_y             = labelled_data
        labelled_train_x, labelled_valid_x = split_data(labelled_x)
        labelled_train_y, labelled_valid_y = split_data(labelled_y)
        
        self.train_classifier_loader = DataLoader(
            Pixel(labelled_train_x, labelled_train_y),
            batch_size,
            shuffle = True
        )
        self.valid_classifier_loader = DataLoader(
            Pixel(labelled_valid_x, labelled_valid_y),
            batch_size
        )
        self.train_autoencoder_loader = DataLoader(
            Pixel(numpy.concatenate((labelled_train_x, unlabelled_train_data))),
            batch_size * 4,
            shuffle = True
        )
        self.valid_autoencoder_loader = DataLoader(
            Pixel(numpy.concatenate((labelled_valid_x, unlabelled_valid_data))),
            batch_size * 4
        )
        
        self.classifier_criterion  = nn.CrossEntropyLoss()
        self.autoencoder_criterion = nn.MSELoss()
        
        self.classifier_optimizer  = optim.Adam(self.classifier.parameters())
        self.autoencoder_optimizer = optim.Adam(self.autoencoder.parameters())
            
    def classifier_step(self, train = True):
        if train:
            self.classifier.train()
            no_grad = Decoy
        else:
            self.classifier.eval()
            no_grad = torch.no_grad
            
        loss_sum      = 0
        correct_count = 0
        with no_grad():
            for x, y in tqdm(
                self.train_classifier_loader if train else self.valid_classifier_loader,
                desc = 'Training classifier' if train else 'Validating classifier'
            ):
                x = x.to(self.device)
                y = y.to(self.device)
                
                prediction        = self.classifier(x)
                loss:torch.Tensor = self.classifier_criterion(prediction, y)
                
                loss_sum      += loss.item()
                correct_count += torch.count_nonzero(torch.argmax(prediction, dim = 1) == y) / len(y)
                
                if train:
                    self.classifier_optimizer.zero_grad()
                    loss.backward()
                    self.classifier_optimizer.step()
                    
        batch_count = len(self.train_classifier_loader if train else self.valid_classifier_loader)
        return loss_sum / batch_count, correct_count / batch_count
    
    def autoencoder_step(self, train = True):
        if train:
            self.autoencoder.train()
            no_grad = Decoy
        else:
            self.autoencoder.eval()
            no_grad = torch.no_grad
            
        loss_sum = 0
        with no_grad():
            for x in tqdm(
                self.train_autoencoder_loader if train else self.valid_autoencoder_loader,
                desc = 'Training autoencoder' if train else 'Validating autoencoder'
            ):
                x = x.to(self.device)

                y                 = self.autoencoder(x)
                loss:torch.Tensor = self.autoencoder_criterion(y, x)
                
                loss_sum += loss.item()
                
                if train:
                    self.autoencoder_optimizer.zero_grad()
                    loss.backward()
                    self.autoencoder_optimizer.step()
                    
        batch_count = len(self.train_autoencoder_loader if train else self.valid_autoencoder_loader)
        return loss_sum / batch_count
    
    def run(self, max_epochs = 100):
        for epoch in tqdm(range(1, max_epochs + 1), desc = 'Epochs'):
            classifier_train_loss, train_accuracy = self.classifier_step()
            classifier_valid_loss, valid_accuracy = self.classifier_step(train = False)
            
            autoencoder_train_loss = self.autoencoder_step()
            autoencoder_valid_loss = self.autoencoder_step(train = False)
            
            tqdm.write(
                f'Epoch {epoch:>3}: ' \
                # f'classifier_train_loss = {classifier_train_loss:>.7f}, ' \
                # f'classifier_train_accuracy = {train_accuracy:>.7f}, ' \
                f'classifier_valid_loss = {classifier_valid_loss:>.7f}, ' \
                f'classifier_valid_accuracy = {valid_accuracy:>.7f}, ' \
                # f'autoencoder_train_loss = {autoencoder_train_loss:>.7f}, ' \
                f'autoencoder_valid_loss = {autoencoder_valid_loss:>.7f}'
            )
            
def load_data(n = 100000):
    with numpy.load('data/1x1_unlabelled_set.npz') as f:
        unlabelled_data = f['x'].astype(numpy.float32)
        
    numpy.random.shuffle(unlabelled_data)
    unlabelled_data = unlabelled_data[:n]
    
    labelled_data_x, labelled_data_y = load_test_data()
    labelled_data_y = labelled_data_y.astype(numpy.int64)
    
    return unlabelled_data, (labelled_data_x, labelled_data_y)
          
def train():
    channels   = 21
    features   = 256
    batch_size = 128
    device     = 'cpu'
    
    encoder    = Encoder(channels, features)
    decoder    = Decoder(features, channels)
    
    autoencoder = Autoencoder(encoder, decoder)
    classifier  = Classifier(encoder, features)
    
    summary(classifier, (channels,))
    summary(autoencoder, (channels,))
    
    unlabelled_data, labelled_data = load_data()
    print(unlabelled_data.shape)
    print(labelled_data[0].shape)
    print(labelled_data[1].shape)
    
    trainer = Trainer(classifier, autoencoder, unlabelled_data, labelled_data, batch_size, device)
    trainer.run()
    
    return classifier

def infer(classifier):
    for image_path in tqdm(IMAGE_PATHS, desc = 'Rollout'):
        reflectance = get_reflectance(os.path.join('data/', image_path))
        
        prediction = numpy.asarray([
            numpy.where(
                numpy.argmax(classifier(torch.tensor(row)).detach().numpy(), axis = 1),
                255, 0
            ) for row in tqdm(reflectance, desc = 'Row')
        ])
        cv2.imwrite(f'rollout/ae_supervised/{image_path}.png', prediction)
            
if __name__ == '__main__':
    classifier = train()
    infer(classifier)