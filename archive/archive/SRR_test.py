import torch
from torch import nn, optim
from torch.utils import data
import numpy
from tqdm import tqdm
from sklearn.svm import OneClassSVM

from utils.load_data import load_train_data, load_test_data

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
        hidden_features = 64
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
            
class SRR:
    def __init__(
        self,
        train_x,
        ensemble_count,
        anomaly_percentage,
        device
    ):
        self.REFINE_EPOCH = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
        
        self.train_x            = train_x
        self.ensemble_count     = ensemble_count
        self.anomaly_percentage = anomaly_percentage
        
        self.representor         = AutoEncoder(train_x.shape[-1])
        self.representor_trainer = AutoEncoderTrainer(self.representor, device)
        
        self.ensemble_classifiers = [
            OneClassSVM(gamma='auto') for _ in range(ensemble_count)
        ]
        
        self.final_classifier = OneClassSVM(gamma='auto', verbose=True)
        
    def calculate_anomaly_percentage(
        self,
        ensemble_classifier,
        anomaly_threshold
    ):
        return numpy.count_nonzero(
            ensemble_classifier.score_samples(self.train_x) > anomaly_threshold
        ) / len(self.train_x)
        
    def calculate_anomaly_threshold(
        self,
        ensemble_classifier,
        eps=0.0001
    ):
        left  = -1e10
        right = 1e10
        while right - left > eps:
            mid = (left + right) / 2
            if self.calculate_anomaly_percentage(ensemble_classifier, mid) < self.anomaly_percentage:
                right = mid
            else:
                left = mid
                
        return right
    
    def calculate_new_labels(
        self,
        anomaly_thresholds
    ):
        labels = []
        for data in tqdm(self.train_x, desc='Labelling'):
            label = 0
            for ensemble_classifier, anomaly_threshold in zip(self.ensemble_classifiers, anomaly_thresholds):
                if ensemble_classifier.score_samples([data]) > anomaly_threshold:
                    label = 1
                    break
            labels.append(label)
            
        return numpy.array(labels)
        
    def refine(self):
        numpy.random.shuffle(self.train_x)
        ensembles = numpy.split(self.train_x, self.ensemble_count)
        
        anomaly_thresholds = []
        for ensemble, ensemble_classifier in tqdm(list(zip(ensembles, self.ensemble_classifiers)), desc='Ensemble'):
            ensemble_classifier.fit(ensemble)
            anomaly_thresholds.append(self.calculate_anomaly_threshold(ensemble_classifier))
            
        return self.train_x[numpy.where(0 == self.calculate_new_labels(anomaly_thresholds))[0]]
    
    def calculate_losses(
        self,
        train_x
    ):
        self.representor.eval()
        
        train_loader = data.DataLoader(IceLeads(train_x))
        losses       = []
        for x in tqdm(train_loader, desc='Calculating loss'):
            prediction = self.representor(x)
            loss       = self.representor_trainer.criterion(prediction, x)
            
            losses.append([loss.item()])
        
        return numpy.array(losses)
        
    def train(
        self,
        batch_size = 32,
        max_epochs = 1000,
        terminate_threshold = 10
    ):
        min_loss        = float('inf')
        terminate_count = 0
        for epoch in tqdm(range(1, max_epochs + 1), desc='SRR'):
            if epoch in self.REFINE_EPOCH:
                train_x = self.refine()
                test(train_x)
                
            train_loader = data.DataLoader(
                IceLeads(train_x),
                batch_size=batch_size,
                shuffle=True,
                pin_memory=True
            )
            
            representor_loss = self.representor_trainer.step(train_loader)
            
            terminate_count += 1
            if representor_loss < min_loss:
                min_loss        = representor_loss
                terminate_count = 0
            elif terminate_count > terminate_threshold:
                break
            
            tqdm.write(f'Epoch {epoch:>3}: train_size={train_x.shape[0]}, loss={representor_loss:>.7f}, min_loss={min_loss:>.7f}')
        
        losses = self.calculate_losses(self.train_x)
        self.final_classifier.fit(losses)
        
def test(train_x):
    global test_x, test_y
    
    i = []
    for x in train_x:
        i.append(numpy.argwhere(x == test_x)[0][0])
        
    i = numpy.array(i)
    tqdm.write(f'{numpy.count_nonzero(test_y[i]) / test_y.shape[0]}')

if __name__ == '__main__':
    device             = 'cpu'
    ensemble_count     = 5
    anomaly_percentage = 0.5
    train_size         = 50000
    
    train_x = load_train_data()
    numpy.random.shuffle(train_x)
    train_x = train_x[:train_size]
    
    global test_x, test_y
    test_x, test_y = load_test_data()
    
    # trainer = SRR(train_x, ensemble_count, anomaly_percentage, device)
    trainer = SRR(test_x, ensemble_count, anomaly_percentage, device)
    trainer.train()
    
    
    losses         = trainer.calculate_losses(test_x)
    
    prediction = numpy.where(
        -1 == trainer.final_classifier.predict(losses),
        1,
        0
    )
    
    accuracy = numpy.count_nonzero(prediction == test_y) / test_y.shape[0]
    print(f'Accuracy: {accuracy}')
    