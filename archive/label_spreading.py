import numpy
import torch
from torch import nn, optim
from torch.utils import data
from sklearn.semi_supervised import LabelSpreading
import cv2
import os
from tqdm import tqdm

from utils.load_data import load_train_data, load_test_data
from utils.split_data import split_data
from utils.get_reflectance import get_reflectance

numpy.seterr(invalid='ignore')

DATA_PATH = 'data'
IMAGE_PATHS = [
    '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'
]
SAVE_PATH = 'label_spreading_map'

class IceLeads(data.Dataset):
    def __init__(self, x):
        self.x = x

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, i):
        return self.x[i]
    
class Encoder(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features = 32
    ):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.BatchNorm1d(hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, hidden_features // 4),
            nn.BatchNorm1d(hidden_features // 4),
            nn.ReLU()
        )
        
    def forward(self, x):
        return self.model(x)
    
    def get_out_features(self):
        return self.model[-2].bias.shape
    
class Decoder(nn.Module):
    def __init__(
        self,
        in_features,
        out_features
    ):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(in_features, in_features * 4),
            nn.BatchNorm1d(in_features * 4),
            nn.ReLU(),
            nn.Linear(in_features * 4, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU()
        )
        
    def forward(self, x):
        return self.model(x)
    
class AutoEncoder(nn.Module):
    def __init__(
        self,
        encoder,
        decoder
    ):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, x):
        return self.decoder(self.encoder(x))
    
    def get_in_features(self):
        return self.encoder.model[0].weight.shape[1:]
    
class AutoEncoderTrainer:
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
            
        return loss_sum / train_loader.batch_size / len(train_loader)
    
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
        max_epochs = 100,
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
            
            tqdm.write(f'Epoch {epoch:>3}: train_loss={train_loss:>.7f}, test_loss={test_loss:>.7f}, min_loss={min_loss:>.7f}')
    
def train_encoder(
    represent_train_x,
    train_ratio,
    batch_size,
    device
):
    encoder      = Encoder(represent_train_x.shape[-1])
    decoder      = Decoder(encoder.get_out_features()[0], represent_train_x.shape[-1])
    auto_encoder = AutoEncoder(encoder, decoder)
    
    trainer = AutoEncoderTrainer(auto_encoder, device)
    
    represent_train_x, represent_test_x = split_data(represent_train_x, train_ratio)
    represent_train_loader = data.DataLoader(
        IceLeads(represent_train_x),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True
    )
    represent_test_loader = data.DataLoader(
        IceLeads(represent_test_x),
        batch_size=batch_size,
        drop_last=True
    )
    
    trainer.fit(represent_train_loader, represent_test_loader)
    
    return encoder

def map_over(lp, encoder, path):
    reflectance = get_reflectance(path)
    
    prediction  = numpy.array([
        lp.predict(
            encoder(torch.tensor(row)).detach().numpy()
        ) for row in tqdm(reflectance)
    ])
    
    return prediction

def save_map(lp, encoder, file_name):
    image = map_over(lp, encoder, os.path.join(DATA_PATH, file_name))
    cv2.imwrite(
        os.path.join(SAVE_PATH, file_name + '_ls_enc.png'),
        image * 255
    )
    
    print(f'Saved: {os.path.join(SAVE_PATH, file_name + ".png")} {image.shape}')

if __name__ == '__main__':
    device      = 'cpu'
    batch_size  = 2048
    train_ratio = 0.8
    alpha       = 0.01
    
    unlabelled_x      = load_train_data()
    represent_train_x = unlabelled_x[10000:10000000]
    unlabelled_x      = unlabelled_x[:10000]
    unlabelled_y = numpy.full(len(unlabelled_x), -1)
    
    labelled_x, labelled_y = load_test_data()
    labelled_x, test_x     = split_data(labelled_x, train_ratio)
    labelled_y, test_y     = split_data(labelled_y, train_ratio)
    
    label_train_x = numpy.concatenate((unlabelled_x, labelled_x))
    label_train_y = numpy.concatenate((unlabelled_y, labelled_y))
    
    encoder = train_encoder(represent_train_x, train_ratio, batch_size, device)
    
    ls = LabelSpreading(kernel='knn', n_jobs=-1, alpha=alpha)
    ls.fit(
        encoder(torch.tensor(label_train_x)).detach().numpy(),
        label_train_y
    )
    
    prediction = ls.predict(
        encoder(torch.tensor(test_x)).detach().numpy()
    )
    print(numpy.count_nonzero(prediction == test_y) / len(test_y))
    
    for file_name in IMAGE_PATHS:
        save_map(ls, encoder, file_name)
    