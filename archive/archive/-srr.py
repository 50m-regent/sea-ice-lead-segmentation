import numpy
from torch import nn, optim
from torch.utils import data
from tqdm import tqdm

class FeatureExtractor(nn.Module):
    def __init__(self, input_features, hidden_features=512):
        super().__init__()
        
        self.f = nn.Sequential(
            nn.Linear(input_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, hidden_features)
        )
        
    def forward(self, input):
        return self.f(input)
    
    @property
    def output_features(self):
        return self.f[-1].bias.shape[0]

class SRR:
    class Dataset(data.Dataset):
        def __init__(self, x, y=None):
            self.x = x
            self.y = numpy.zeros(x.shape[0], 1) if None is y else y
    
        def __len__(self):
            return len(self.x)
        
        def __getitem__(self, i):
            return self.x[i], self.y[i]
    
    class OCC(nn.Module):
        def __init__(self, input_features, hidden_features=512):
            super().__init__()
            
            self.f = nn.Sequential(
                nn.Linear(input_features, hidden_features),
                nn.ReLU(),
                nn.Linear(hidden_features, hidden_features),
                nn.ReLU(),
                nn.Linear(hidden_features, 1)
            )
            
        def forward(self, input):
            return self.f(input)
    
    def __init__(
        self,
        train_x,
        test_x,
        test_y,
        feature_extractor,
        ensemble_count,
        anomaly_percentile_threshold,
        device
    ):
        self.train_x                      = train_x
        self.ensemble_count               = ensemble_count
        self.anomaly_percentile_threshold = anomaly_percentile_threshold
        self.device                       = device
        
        self.feature_extractor           = feature_extractor
        self.feature_extractor_criterion = nn.
        
        self.refiners   = [self.OCC(self.feature_extractor.output_features) for _ in range(ensemble_count)]
        self.criterions = [nn.BCELoss() for _ in range(ensemble_count)]
        self.optimizers = [optim.Adam(refiner.parameters()) for refiner in self.refiners]
        
    def train_occ_step(
        self,
        occ,
        train_loader,
        criterion,
        optimizer
    ):
        occ.train()
        
        loss_sum      = 0
        correct_count = 0
        for x, y in tqdm(train_loader, desc='Training'):
            x, y = x.to(self.device), y.to(self.device)
            
            prediction = occ(x)
            loss       = criterion(prediction, y)
            
            loss_sum      += loss.item()
            correct_count += numpy.count_nonzero(abs(prediction - y) < 0.5)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        tqdm.write(f'Train loss: {loss_sum / len(train_loader):>0.5f}, Train accuracy: {correct_count / train_loader.batch_size / len(train_loader):>0.5f}')
        return loss_sum / len(train_loader)
        
    def train_occ(
        self,
        occ,
        train_x,
        criterion,
        optimizer,
        terminate_threshold=10
    ):
        train_loader = data.DataLoader(self.Dataset(self.feature_extractor(train_x)), batch_size=256, shuffle=True, pin_memory=True)
        
        min_loss        = float('inf')
        terminate_count = 0
        while True:
            loss = self.train_occ_step(occ, train_loader, criterion, optimizer)
            
            if loss < min_loss:
                min_loss = loss
                terminate_count = 0
                
            terminate_count += 1
            if terminate_count > terminate_threshold:
                break
            
    def calculate_anomaly_percentage(self, k, anomaly_threshold):
        true_count = 0
        for data in tqdm(self.train_x, desc='Calculating anomaly percentage'):
            if self.refiners[k](self.feature_extractor(numpy.array([data]))) < anomaly_threshold:
                continue
            
            true_count += 1
            
        return true_count / len(self.train_x)
            
    def calculate_anomaly_threshold(self, k, eps=0.0001):
        left  = -1e10
        right = 1e10
        while right - left > eps:
            mid = (left + right) / 2
            if self.calculate_anomaly_percentage(k, mid) < self.anomaly_percentile_threshold:
                right = mid
            else:
                left = mid
                
        return right
    
    def calculate_new_labels(self, anomaly_thresholds):
        labels = []
        for data in tqdm(self.train_x, desc='Calculating new_labels'):
            label = 0
            for k in range(self.ensemble_count):
                if self.refiners[k](self.feature_extractor(numpy.array([data]))) > anomaly_thresholds:
                    label = 1
                    break
            labels.append(label)
            
        return numpy.array(label)
        
    def refine(self):
        numpy.random.shuffle(self.train_x)
        ensembles = numpy.split(self.train_x, self.ensemble_count)
        
        anomaly_thresholds = []
        for k, (occ, ensemble) in enumerate(zip(self.refiners, ensembles)):
            self.train_occ(occ, ensemble, self.criterions[k], self.optimizers[k])
            anomaly_thresholds.append(self.calculate_anomaly_threshold(k))
            
        return self.train_x[numpy.where(0 == self.calculate_new_labels(anomaly_thresholds))]
    
    def train_feature_extractor_step(self, train_loader):
        self.feature_extractor.train()
        
        loss_sum      = 0
        for x, y in tqdm(train_loader, desc='Training'):
            x, y = x.to(self.device), y.to(self.device)
            
            prediction = self.feature_extractor(x)
            loss       = self.feature_extractor_criterion(prediction, y)
            loss_sum  += loss.item()
            
            self.feature_extractor_optimizer.zero_grad()
            loss.backward()
            self.feature_extractor_optimizer.step()
            
        return loss_sum / len(train_loader)
    
    def train_feature_extractor(self, train_loader, terminate_threshold=10):
        min_loss        = float('inf')
        terminate_count = 0
        while True:
            loss = self.train_feature_extractor_step(train_loader)
            
            if loss < min_loss:
                min_loss = loss
                terminate_count = 0
                
            terminate_count += 1
            if terminate_count > terminate_threshold:
                break
    
    def train(self, terminate_threshold=10):
        min_loss        = float('inf')
        terminate_count = 0
        while True:
            train_x = self.refine()
            
            train_loader = data.DataLoader(self.Dataset(train_x), batch_size=1024, shuffle=True, pin_memory=True)
            loss         = self.train_feature_extractor(train_loader)
            
            if loss < min_loss:
                min_loss = loss
                terminate_count = 0
            
            terminate_count += 1
            if terminate_count > terminate_threshold:
                break
            
        final_occ = self.OCC(self.feature_extractor.output_features)
        train_x   = self.refine()
        criterion = nn.BCELoss()
        optimizer = optim.Adam(final_occ.parameters())
        self.train_occ(final_occ, train_x, criterion, optimizer)
        
        return final_occ
        

def load_train_data():
    with numpy.load('data/all_reflectance.npz') as f:
        return f['arr_0'].astype(numpy.float32)

def load_test_data():
    with numpy.load('data/1x1_test_set.npz') as f:
        return f['x'].astype(numpy.float32), f['y'].astype(numpy.float32)

if __name__ == '__main__':
    device                       = 'cpu'
    ensemble_count               = 5
    anomaly_percentile_threshold = 0.5
    
    train_x        = load_train_data()
    test_x, test_y = load_test_data()
    
    trainer = SRR(train_x, test_x, test_y, ensemble_count, anomaly_percentile_threshold, device)
    occ = trainer.train()
    