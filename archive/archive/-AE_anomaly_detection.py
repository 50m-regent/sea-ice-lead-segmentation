from torch import nn, optim
from torch.utils import data
from torchvision.datasets import MNIST
import numpy
from tqdm import tqdm

class IceLeads(data.Dataset):
    def __init__(self, x):
        self.x = x

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, i):
        return self.x[i]

class AutoEncoder(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features = 512
    ):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, hidden_features // 2),
            nn.BatchNorm1d(hidden_features // 2),
            nn.Linear(hidden_features // 2, hidden_features // 4),
            nn.BatchNorm1d(hidden_features // 4),
            nn.Linear(hidden_features // 4, hidden_features // 8)
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_features // 8, hidden_features // 4),
            nn.BatchNorm1d(hidden_features // 4),
            nn.Linear(hidden_features // 4, hidden_features // 2),
            nn.BatchNorm1d(hidden_features // 2),
            nn.Linear(hidden_features // 2, hidden_features),
            nn.BatchNorm1d(hidden_features),
            nn.Linear(hidden_features, in_features),
            nn.ReLU(),
        )
    
    def forward(self, x):
        return self.decoder(self.encoder(x))
    
class AutoEncoderTrainer:
    def __init__(
        self,
        model,
        device
    ):
        self.model  = model.to(device)
        self.device = device
        
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters())
        
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
    
    def get_losses(
        self,
        test_loader
    ):
        self.model.eval()
        
        losses = []
        for x in tqdm(test_loader, desc='Testing'):
            x = x.to(self.device)
            
            prediction = self.model(x)
            loss       = self.criterion(prediction, x)
            
            losses.append(loss.item())
            
        return numpy.array(losses)
    
    @staticmethod
    def calculate_accuracy(
        threshold,
        losses,
        test_y
    ):
        return numpy.count_nonzero(
            numpy.where(losses > threshold, 1, 0) == test_y
        ) / test_y.shape[0]
    
    @staticmethod
    def calculate_best_accuracy(
        losses,
        test_y
    ):
        return max(
            [AutoEncoderTrainer.calculate_accuracy(loss, losses, test_y) for loss in tqdm(losses, desc='Calculating accuracy')]
        )
        
    def fit(
        self,
        train_loader,
        test_loader,
        test_y,
        max_epochs = 500,
        terminate_threshold = 10
    ):  
        min_loss        = float('inf')
        terminate_count = 0
        for epoch in tqdm(range(max_epochs), desc='Fitting'):
            loss = self.step(train_loader)
            
            terminate_count += 1
            if loss < min_loss:
                min_loss        = loss
                terminate_count = 0
            elif terminate_count > terminate_threshold:
                break
            
            losses   = self.get_losses(test_loader)
            accuracy = AutoEncoderTrainer.calculate_best_accuracy(losses, test_y)
            tqdm.write(f'Epoch {epoch:>3}: loss={loss:>.7f}, min_loss={min_loss:>.7f}, accuracy={accuracy}')

if __name__ == '__main__':
    device     = 'cpu'
    batch_size = 32
    
    mnist = MNIST('data/MNIST', download=True)
    train_x = []
    train_y = []
    test_x  = []
    test_y  = []
    for x, y in mnist:
        if 0 == y:
            train_x.append(numpy.array(x).reshape(-1, ))
            train_y.append(0)
        elif 5 == y:
            test_x.append(numpy.array(x).reshape(-1, ))
            test_y.append(1)
            
    train_x = numpy.array(train_x, dtype=numpy.float32)
    train_y = numpy.array(train_y, dtype=numpy.float32)
    test_x  = numpy.array(test_x, dtype=numpy.float32)
    test_y  = numpy.array(test_y, dtype=numpy.float32)
    
    test_x = numpy.concatenate((test_x, train_x[4000:]))
    test_y = numpy.concatenate((test_y, train_y[4000:]))
    train_x = train_x[:4000]
    train_y = train_y[:4000]
    
    auto_encoder = AutoEncoder(train_x.shape[-1])
    trainer      = AutoEncoderTrainer(auto_encoder, device)
    
    train_loader = data.DataLoader(
        IceLeads(train_x),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True
    )
    test_loader = data.DataLoader(IceLeads(test_x))
    trainer.fit(train_loader, test_loader, test_y)
    