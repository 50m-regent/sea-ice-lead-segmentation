import numpy
import os
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import cv2

from utils.get_reflectance import get_reflectance

DATA_PATH = 'data'
FILE_PATH = [
    'S3B_OL_1_EFR____20190301T232521_20190301T232821_20200111T235148_0179_022_301_1800_MR1_R_NT_002.SEN3',
    'S3A_OL_1_EFR____20180307T054004_20180307T054119_20180308T091959_0075_028_319_1620_LN1_O_NT_002.SEN3'
]
TEST_PATH  = '1x1_test_set.npz'

class Model(nn.Module):
    def __init__(self, in_size, out_size, hidden_size=512):
        super(Model, self).__init__()
        
        self.dense_stack = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_size),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.dense_stack(x)
    
class IceLeads(Dataset):
    def __init__(self, x, y):
        super(IceLeads, self).__init__()
            
        self.data = torch.tensor(x)
        self.label = torch.tensor(y)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        return self.data[i], self.label[i]
    
class Trainer:
    def __init__(self, train_loader, test_loader, device):
        self.train_loader = train_loader
        self.test_loader  = test_loader
        
        self.device = device
        
        self.model = Model(
            len(self.train_loader.dataset[0][0]),
            len(self.train_loader.dataset[0][1])
        ).to(self.device)
        
        self.loss_function = nn.MSELoss()
        self.optimizer     = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
    def train_step(self):
        for x, y in tqdm(self.train_loader, desc='Training'):
            x, y = x.to(self.device), y.to(self.device)
            
            prediction = self.model(x)
            loss       = self.loss_function(prediction, y)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
    def test_step(self):
        size = len(self.test_loader.dataset)
        loss = 0

        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)
                
                prediction = self.model(x)
                loss += self.loss_function(prediction, y).item() / size
                
            tqdm.write(f'Loss: {loss:>0.7f}, ', end='')
            
        return loss
        
    def train(self, epoch, threshold=10):
        min_loss = float('inf')
        unsurpass_count = 0
        
        for e in range(epoch):
            self.train_step()
            loss = self.test_step()
            
            unsurpass_count += 1
            if min_loss > loss:
                unsurpass_count = 0
                min_loss = loss
                
            if unsurpass_count >= threshold:
                tqdm.write('Threshold reached')
                break
            
            tqdm.write(f'Epoch: {e + 1}/{epoch}')

if __name__ == '__main__':
    # device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    device = 'cpu'
    epoch  = 100
    
    with numpy.load(os.path.join(DATA_PATH, TEST_PATH)) as data:
        x = data['x'].astype('float32')
        y = data['y'].reshape(-1, 1).astype('float32')
        
    print(x)
    print(y)
        
    train_loader = DataLoader(IceLeads(x[:4000], y[:4000]), batch_size=32, pin_memory=True, shuffle=True)
    test_loader  = DataLoader(IceLeads(x[4000:], y[4000:]), pin_memory=True)
    
    trainer = Trainer(train_loader, test_loader, device)
    trainer.train(epoch)
    
    for file in FILE_PATH:
        image = get_reflectance(os.path.join(DATA_PATH, file)).astype('float32')[::2, ::2]
        image_data = image.reshape(-1, image.shape[2])
        print(image_data.shape)
        print(image_data)
        
        image_loader = DataLoader(IceLeads(image_data, numpy.zeros(len(image_data))))
        predicted = numpy.array([
            trainer.model(x.to(device)).detach().numpy() for x, _ in tqdm(image_loader, desc=f'{file}')
        ])
        predicted = predicted.reshape(image.shape[:2]) * 255
        
        cv2.imwrite(f'labels_test/{file}.png', predicted)
        
        print(predicted)
    