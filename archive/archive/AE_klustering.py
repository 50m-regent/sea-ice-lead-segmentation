import torch
from torch import nn, optim
from torch.utils import data
import torchsummary
from tqdm import tqdm
from sklearn.svm import OneClassSVM
import numpy

from utils.load_data import load_train_data, load_test_data

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
            nn.Linear(hidden_features, hidden_features),
            nn.BatchNorm1d(hidden_features),
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
            nn.Linear(in_features, in_features),
            nn.BatchNorm1d(in_features),
            nn.ReLU(),
            nn.Linear(in_features, out_features),
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
        
        torchsummary.summary(self.model, self.model.get_in_features())
        
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
            elif terminate_count > terminate_threshold and train_loss < min_loss:
                break
            
            tqdm.write(f'Epoch {epoch:>3}: loss={train_loss:>.7f}, test_loss={test_loss:>.7f}, min_loss={min_loss:>.7f}')

if __name__ == '__main__':
    device     = 'cpu'
    batch_size = 512
    
    train_x        = load_train_data()
    test_x, test_y = load_test_data()
    
    encoder      = Encoder(train_x.shape[-1])
    decoder      = Decoder(encoder.get_out_features()[0], train_x.shape[-1])
    auto_encoder = AutoEncoder(encoder, decoder)
    
    trainer = AutoEncoderTrainer(auto_encoder, device)
    
    train_loader = data.DataLoader(
        IceLeads(train_x[:1000000]),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True
    )
    test_loader = data.DataLoader(
        IceLeads(train_x[-500000:]),
        batch_size=batch_size
    )
    validation_loader = data.DataLoader(IceLeads(test_x))
    
    trainer.fit(train_loader, test_loader, max_epochs=30)
    
    validation_loss = trainer.test(validation_loader)
    print(f'Validation loss: {validation_loss:>.7f}')
    
    train_loader = data.DataLoader(IceLeads(train_x[:100000]))
    train_features = numpy.array([
        encoder(x.to(device)).detach().numpy()[0] for x in train_loader
    ])
    validation_features = numpy.array([
        encoder(x.to(device)).detach().numpy()[0] for x in validation_loader
    ])
    
    svm = OneClassSVM(gamma='auto', verbose=True).fit(train_features)
    # svm = OneClassSVM(gamma='auto', verbose=True).fit(validation_features)
    
    prediction = numpy.where(-1 == svm.predict(validation_features), 1, 0)
    accuracy   = numpy.count_nonzero(prediction == test_y) / test_y.shape[0]
    print(f'Accuracy: {accuracy:>.7f}')
    