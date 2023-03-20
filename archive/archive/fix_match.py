import numpy
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import random

def load_train_data():
    with numpy.load('data/all_reflectance.npz') as f:
        return f['arr_0'].astype(numpy.float32)

def load_test_data():
    with numpy.load('data/1x1_test_set.npz') as f:
        return f['x'].astype(numpy.float32), f['y'].astype(numpy.float32)
    
class IceLeads(Dataset):
    def __init__(self, x, y, expansion=False):
        self.x = x
        self.y = y
        
        if expansion:
            self.x = numpy.concatenate((
                self.x,
                numpy.array([
                    a + random.gauss(0, 0.1) for a in x
                ])
            ))
            self.y = numpy.concatenate((self.y, y))
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, i):
        return self.x[i], self.y[i]

class Model(nn.Module):
    def __init__(self, in_features, hidden_features=512):
        super().__init__()
        
        self.f = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, 1),
            nn.Sigmoid()
        )
        
    def forward(self, input):
        return self.f(input)
    
class Ingest:
    def __init__(self, unlabelled_x, labelled_x, labelled_y, device, learning_rate, batch_size=64):
        self.device     = device
        self.batch_size = batch_size
        
        self.unlabelled_x = unlabelled_x
        
        self.train_x = labelled_x[4000:]
        self.train_y = labelled_y[4000:]
        
        self.train_loader = DataLoader(
            IceLeads(self.train_x, self.train_y, expansion=True),
            batch_size=8,
            shuffle=True,
            pin_memory=True
        )
        self.test_loader = DataLoader(IceLeads(labelled_x[:4000], labelled_y[:4000]), pin_memory=True)
        
        self.model      = Model(unlabelled_x.shape[-1]).to(device)
        self.criterion  = nn.BCELoss()
        self.optimizer  = optim.SGD(self.model.parameters(), lr=learning_rate)
        
    def create_loader(self):
        self.train_loader = DataLoader(
            IceLeads(self.train_x, self.train_y),
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True
        )
        
    def train_step(self):
        self.model.train()
        
        loss_sum      = 0
        correct_count = 0
        for x, y in tqdm(self.train_loader, desc='Training'):
            x, y = x.to(self.device), y.to(self.device)
            
            prediction = self.model(x)
            loss       = self.criterion(prediction, y)
            
            loss_sum      += loss.item()
            correct_count += numpy.count_nonzero(abs(prediction - y) < 0.5)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        tqdm.write(f'Train loss: {loss_sum / len(self.train_loader):>0.5f}, Train accuracy: {correct_count / self.train_loader.batch_size / len(self.train_loader):>0.5f}')
        return loss_sum / len(self.train_loader)
    
    def test(self):
        self.model.eval()
        
        loss_sum      = 0
        correct_count = 0
        for x, y in self.test_loader:
            x, y = x.to(self.device), y.to(self.device)
            
            prediction = self.model(x)
            loss       = self.criterion(prediction, y)
            
            loss_sum      += loss.item()
            correct_count += 1 if abs(prediction - y) < 0.5 else 0
                
        tqdm.write(f'Test loss: {loss_sum / len(self.test_loader):>0.5f}, Test accuracy: {correct_count / len(self.test_loader):>0.5f}')
        return loss_sum / len(self.test_loader)
        
    def ingest(self):
        self.model.eval()
        unlabelled_prediction = numpy.array([
            self.model(torch.tensor(x)).detach().numpy() for x in tqdm(
                self.unlabelled_x.reshape(-1, 1, self.unlabelled_x.shape[-1]),
                desc='Evaluating unlabelled data'
            )
        ])
        
        newcomer_0index = numpy.where(min(unlabelled_prediction) + 0.05 > unlabelled_prediction)[0]
        newcomer_1index = numpy.where(max(unlabelled_prediction) - 0.05 < unlabelled_prediction)[0]
        
        if 0 == len(newcomer_0index) or 0 == len(newcomer_1index):
            return False
        
        if len(newcomer_0index) > len(newcomer_1index):
            newcomer_0index = newcomer_0index[:len(newcomer_1index)]
        else:
            newcomer_1index = newcomer_1index[:len(newcomer_0index)]
        
        new_x0 = self.unlabelled_x[newcomer_0index]
        new_y0 = numpy.zeros((len(new_x0), 1), dtype=numpy.float32)
        new_x1 = self.unlabelled_x[newcomer_1index]
        new_y1 = numpy.ones((len(new_x1), 1), dtype=numpy.float32)
        
        self.unlabelled_x = numpy.delete(self.unlabelled_x, numpy.concatenate((newcomer_0index, newcomer_1index)), axis=0)
        
        self.train_x = numpy.concatenate((self.train_x, new_x0, new_x1))
        self.train_y = numpy.concatenate((self.train_y, new_y0, new_y1))
        
        tqdm.write(f'Unlabelled data: {self.unlabelled_x.shape}')
        tqdm.write(f'Labelled data: {self.train_x.shape} {self.train_y.shape}, 0 count: {numpy.count_nonzero(0 == self.train_y)}, 1 count: {numpy.count_nonzero(1 == self.train_y)}')
        
        return True
        
    def train(self, num_iteration, max_epochs, threshold=5):
        for iteration in range(1, num_iteration + 1):
            tqdm.write(f'Iteration: {iteration:>3}')
            
            min_loss  = float('inf')
            cut_count = 0
            for epoch in range(1, max_epochs + 1):
                tqdm.write(f'Epoch: {epoch:>4}')
                cut_count += 1
                
                loss = self.train_step()
                # loss = self.test()
                self.test()
                if min_loss > loss:
                    min_loss = loss
                    cut_count = 0
                elif cut_count > threshold:
                    break
                
            if not self.ingest():
                print('No more newcomers')
                break
            
            self.create_loader()

if __name__ == '__main__':
    # device        = 'mps' if torch.backends.mps.is_available() else 'cpu'
    device = 'cpu'
    learning_rate = 0.001
    num_iteration = 10
    max_epochs    = 100
    
    unlabelled_x           = load_train_data()
    labelled_x, labelled_y = load_test_data()
    labelled_y             = numpy.reshape(labelled_y, (-1, 1))
    
    print(f'Unlabelled data: {unlabelled_x.shape}')
    print(f'Labelled data: {labelled_x.shape} {labelled_y.shape}, 0 count: {numpy.count_nonzero(0 == labelled_y)}, 1 count: {numpy.count_nonzero(1 == labelled_y)}')
    
    trainer = Ingest(unlabelled_x, labelled_x, labelled_y, device, learning_rate)
    trainer.train(num_iteration, max_epochs)
    torch.save(trainer.model.state_dict(), 'model/ingest.pt')
    