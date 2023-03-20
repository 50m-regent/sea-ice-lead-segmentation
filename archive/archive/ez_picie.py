import os
import numpy
from tqdm import tqdm
import torch
from torch.nn import Module, Sequential, Conv2d, ReLU, MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torchsummary import summary
from sklearn.cluster import KMeans

from utils.split_data import split_data

DATA_PATH = 'data/trimmed256_1.npz'

class Images(Dataset):
    def __init__(self, x):
        self.x = x
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, i):
        return self.x[i]

class Encoder(Module):
    def __init__(self, in_channels, features = 4):
        super().__init__()
        
        self.features = features
        
        self.model = Sequential(
            Conv2d(in_channels, out_channels = 64, kernel_size = 5, padding = 2),
            ReLU(),
            
            Conv2d(in_channels = 64, out_channels = features, kernel_size = 3, padding = 1),
            ReLU()
        )
        
    def forward(self, x):
        return self.model(x)
    
class Trainer:
    def __init__(self, images:numpy.ndarray, encoder:Encoder, device:str, train_ratio = 0.8, batch_size = 128, n_clusters = 3):
        train_images, valid_images = split_data(images, train_ratio)
        self.train_loader = DataLoader(
            Images(train_images),
            batch_size,
            shuffle = True
        )
        self.valid_loader = DataLoader(
            Images(valid_images),
            batch_size
        )
        
        self.encoder    = encoder.to(device)
        self.device     = device
        self.n_clusters = n_clusters
        
        self.criterion = MSELoss()
        self.optimizer = Adam(self.encoder.parameters())
        
    def fit_kmeans(self) -> KMeans:
        kmeans = KMeans(self.n_clusters, n_init = 'auto', verbose = True)
        
        all_features = []
        for x in self.train_loader:
            features = self.encoder(x).detach().numpy().reshape(-1, self.encoder.features)[:100]
            all_features.append(features)
            
        all_features = numpy.concatenate(all_features)
        kmeans.fit(all_features)
        
        return kmeans
    
    def get_labels(self, x:torch.tensor) -> torch.tensor:
        assert hasattr(self, 'kmeans')
        
        features = self.encoder(x).detach().numpy()
        
        all_labels = []
        for feature in features:
            prediction = self.kmeans.predict(feature.reshape(-1, self.encoder.features))
            
            labels = self.kmeans.cluster_centers_[prediction].reshape(feature.shape)
            all_labels.append(labels)
            
        all_labels = numpy.array(all_labels, dtype = numpy.float32)
                
        return torch.tensor(all_labels)
        
    def train_step(self) -> float:
        self.encoder.train()
        
        self.kmeans   = self.fit_kmeans()
        loss_sum = 0
        for x in tqdm(self.train_loader, desc = 'Training'):
            x = x.to(self.device)
            
            features = self.encoder(x)
            labels   = self.get_labels(x)
            loss     = self.criterion(features, labels)
            
            loss_sum += loss.item()
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        return loss_sum / len(self.train_loader)
        
    def valid_step(self) -> float:
        self.encoder.eval()
        
        loss_sum = 0
        with torch.no_grad():
            for x in tqdm(self.valid_loader, desc = 'Validating'):
                x = x.to(self.device)
                
                features = self.encoder(x)
                labels   = self.get_labels(x)
                loss     = self.criterion(features, labels)
                
                loss_sum += loss.item()
        
        return loss_sum / len(self.valid_loader)

    def run(self, max_epochs:int = 100, terminate_threshold:int = 10):
        min_loss        = float('inf')
        terminate_count = 0
        for epoch in tqdm(range(1, max_epochs + 1), desc = 'Epochs'):
            train_loss = self.train_step()
            valid_loss = self.valid_step()
            
            tqdm.write(
                f'Epoch {epoch:>3}: ' \
                f'train_loss = {train_loss:>.7f}, ' \
                f'valid_loss = {valid_loss:>.7f}, ' \
                f'min_loss = {min_loss:>.7f}'
            )
            
            terminate_count += 1
            if min_loss > valid_loss:
                min_loss = valid_loss
            elif terminate_count > terminate_threshold:
                break
        
def get_images() -> numpy.ndarray:
    '''
    files = os.listdir(DATA_PATH)
    files = [file for file in files if '.npz' in file]
    
    images = []
    for file in tqdm(files, desc = 'Loading'):
        with numpy.load(os.path.join(DATA_PATH, file)) as f:
            images.append(f['x'].astype(numpy.float32))
            
    images = numpy.array(images)
    numpy.random.shuffle(images)
    '''
    
    with numpy.load(DATA_PATH) as f:
        images = f['arr_0'].astype(numpy.float32)
        
    images = images.transpose(0, 3, 1, 2)
    
    return images

def main():
    # device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    device = 'cpu'
    
    images = get_images()
    print(images.shape)
    
    encoder = Encoder(in_channels = images.shape[1])
    summary(encoder, images.shape[1:])
    
    trainer = Trainer(images, encoder, device, batch_size = 16)
    trainer.run()
    
    torch.save(encoder, f'model/unsupervised.pt')

if __name__ == '__main__':
    main()