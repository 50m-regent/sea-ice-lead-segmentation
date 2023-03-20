import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchsummary import summary
from tqdm import tqdm
import numpy
import os
import cv2
from sklearn.feature_extraction.image import extract_patches_2d

from utils.get_reflectance import get_reflectance

torch.autograd.set_detect_anomaly(True)

class Embedding(nn.Module):
    def __init__(self, features, n_patches):
        super().__init__()
        
        self.token     = nn.Parameter(torch.randn(1, 1, features))
        self.embedding = nn.Parameter(torch.randn(1, n_patches + 1, features))
        
    def forward(self, x):
        assert len(x.shape) == 3
        
        batch_size, *_ = x.shape
        
        tokens = torch.cat([self.token] * batch_size)
        x = torch.cat((tokens, x), dim = 1)
        
        x += self.embedding
        return x
    
class MultiHeadAttention(nn.Module):
    def __init__(self, features, n_heads):
        super().__init__()
        
        self.n_heads       = n_heads
        self.head_features = features // n_heads
        
        self.W_q = nn.Linear(features, features)
        self.W_k = nn.Linear(features, features)
        self.W_v = nn.Linear(features, features)
        
        self.softmax = nn.Softmax(dim = -1)
        
    def split(self, t):
        t = t.reshape((t.shape[0], t.shape[1], self.n_heads, self.head_features))
        t = t.transpose(1, 2)
        
        return t
    
    def cat(self, t):
        t = t.transpose(1, 2)
        t = t.reshape((t.shape[0], t.shape[1], self.n_heads * self.head_features))
        
        return t
        
    def forward(self, x):
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)
        
        q = self.split(q)
        k = self.split(k)
        v = self.split(v)
        
        logit = torch.matmul(q, k.transpose(-1, -2)) * (self.head_features ** -0.5)
        attention_weight = self.softmax(logit)

        y = self.cat(torch.matmul(attention_weight, v))
        return y
    
class TransformerEncoder(nn.Module):
    def __init__(self, projection_features, n_heads, depth, mlp_features):
        super().__init__()
        
        self.depth = depth
        
        self.norm = nn.LayerNorm(projection_features)
        self.mha  = MultiHeadAttention(projection_features, n_heads)
        self.mlp  = nn.Sequential(
            nn.Linear(projection_features, mlp_features),
            nn.ReLU(),
            nn.Linear(mlp_features, projection_features)
        )
        
    def forward(self, x):
        for _ in range(self.depth):
            x = x + self.mha(self.norm(x))
            x = x + self.mlp(self.norm(x))
            
        return x
    
class Patcher(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        
        self.patching = nn.Unfold(kernel_size = (patch_size, patch_size))
        
    def forward(self, x):
        x = self.patching(x)
        x = x.transpose(1, 2)
        
        return x

class VisionTransformer(nn.Module):
    def __init__(
        self,
        image_size,
        patch_size,
        n_classes,
        projection_features,
        n_heads,
        depth,
        mlp_features
    ):
        super().__init__()
        
        image_channel, image_size, _ = image_size
        
        n_patches       = (image_size // patch_size) ** 2
        patch_dimension = image_channel * patch_size ** 2
        
        self.patcher = Patcher(patch_size)
        self.projecting = nn.Linear(patch_dimension, projection_features)
        self.embedding = Embedding(projection_features, n_patches)
        self.transformer = TransformerEncoder(
            projection_features,
            n_heads,
            depth,
            mlp_features
        )
        
        self.encoder = nn.Sequential(
            self.patcher,
            self.projecting,
            self.embedding,
            self.transformer
        )
        self.head = nn.Sequential(
            nn.LayerNorm(projection_features),
            nn.Linear(projection_features, n_classes)
        )
        
    def forward(self, x):
        x = self.encoder(x)
        y = self.head(x[:, 0])
        
        return y
    
class Decoder(nn.Module):
    def __init__(self, in_features, hidden_features, out_size):
        super().__init__()
        
        self.out_size = out_size
        
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(),
            
            nn.Linear(hidden_features, out_size[2]),
            nn.ReLU()
        )
        self.reshape = nn.Unflatten(1, out_size[:2])
        
    def forward(self, x):
        x = self.mlp(x)
        x = self.reshape(x)
        x = torch.permute(x, (0, 3, 1, 2))
        
        return x
    
class AutoEncoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, x):
        return self.decoder(self.encoder(x)[:, 1:])
    
class Decoy:
    def __enter__(self):
        pass
    
    def __exit__(self, _, __, ___):
        pass
    
class Trainer:
    def __init__(
        self,
        vit, ae,
        train_vit_dataset, valid_vit_dataset,
        train_ae_dataset, valid_ae_dataset,
        batch_size,
        device
    ):
        self.vit:VisionTransformer = vit.to(device)
        self.ae:AutoEncoder        = ae.to(device)
        
        self.train_vit_loader = DataLoader(
            train_vit_dataset,
            batch_size // 4,
            shuffle = True
        )
        self.valid_vit_loader = DataLoader(
            valid_vit_dataset,
            batch_size // 4,
            shuffle = True
        )
        self.train_ae_loader = DataLoader(
            train_ae_dataset,
            batch_size,
            shuffle = True
        )
        self.valid_ae_loader = DataLoader(
            valid_ae_dataset,
            batch_size,
            shuffle = True
        )
        
        self.vit_criterion = nn.CrossEntropyLoss()
        self.ae_criterion  = nn.MSELoss()
        
        self.vit_optimizer = optim.Adam(self.vit.parameters())
        self.ae_optimizer  = optim.Adam(self.ae.parameters())
        
        self.device = device
        
    def vit_step(self, train = True):
        if train:
            self.vit.train()
            no_grad = Decoy
        else:
            self.vit.eval()
            no_grad = torch.no_grad
        
        loss_sum  = 0
        n_correct = 0
        with no_grad():
            for x, y in tqdm(
                self.train_vit_loader if train else self.valid_vit_loader,
                desc = 'Training' if train else 'Validating'
            ):
                x = x.to(self.device)
                y = y.to(self.device)
                
                prediction:torch.Tensor = self.vit(x)
                
                loss:torch.Tensor = self.vit_criterion(prediction, y)
                loss_sum += loss.item()
                n_correct += (prediction.argmax(dim = 1) == y).float().mean()
                
                if train:
                    self.vit_optimizer.zero_grad()
                    loss.backward()
                    self.vit_optimizer.step()
            
        data_count = len(self.train_vit_loader if train else self.valid_vit_loader)
        return loss_sum / data_count, n_correct / data_count
    
    def ae_step(self, train = True):
        if train:
            self.ae.train()
            no_grad = Decoy
        else:
            self.ae.eval()
            no_grad = torch.no_grad
        
        loss_sum = 0
        with no_grad():
            for x  in tqdm(
                self.train_ae_loader if train else self.valid_ae_loader,
                desc = 'Training' if train else 'Validating'
            ):
                x = x.to(self.device)
                y = self.ae(x)
                
                loss:torch.Tensor = self.ae_criterion(y, x)
                loss_sum += loss.item()
                
                if train:
                    self.ae_optimizer.zero_grad()
                    loss.backward()
                    self.ae_optimizer.step()
            
        data_count = len(self.train_ae_loader if train else self.valid_ae_loader)
        return loss_sum / data_count
    
    def run(self, n_epochs = 100):
        for epoch in tqdm(range(1, n_epochs + 1), desc = 'Epochs'):
            train_vit_loss, train_vit_accuracy = self.vit_step()
            valid_vit_loss, valid_vit_accuracy = self.vit_step(train = False)
            
            train_ae_loss = self.ae_step()
            valid_ae_loss = self.ae_step(train = False)
            
            tqdm.write(
                f'Epoch {epoch:>3}: ' \
                f'vit_loss = {valid_vit_loss:>.7f}, ' \
                f'vit_accuracy = {valid_vit_accuracy:>.7f}, ' \
                f'ae_loss = {valid_ae_loss:>.7f}'
            )
            
class UnlabelledPixels(Dataset):
    def __init__(self, path, i):
        with numpy.load(path) as f:
            self.data = f['arr_0'].astype(numpy.float32)[i].transpose(0, 3, 1, 2)
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        return self.data[i]
    
class LabelledPixels(Dataset):
    def __init__(self, path, i = None):
        with numpy.load(path) as f:
            self.x = f['x'].astype(numpy.float32).transpose(0, 3, 1, 2)
            self.y = f['y'].astype(numpy.int64)
            
            if None is i:
                return
            
            self.x = self.x[i]
            self.y = self.y[i]
            
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, i):
        return self.x[i], self.y[i]
    
def train():
    features        = 64
    hidden_features = 128
    batch_size      = 256
    device          = 'cpu'
    
    vit = VisionTransformer(
        image_size = (21, 3, 3),
        patch_size = 1,
        n_classes = 2,
        projection_features = features,
        n_heads = 4,
        depth = 3,
        mlp_features = hidden_features
    )
    decoder = Decoder(features, hidden_features, out_size = (3, 3, 21))
    ae = AutoEncoder(vit.encoder, decoder)
    
    summary(vit, (21, 3, 3))
    summary(ae, (21, 3, 3))
    
    
    train_vit_dataset = LabelledPixels('data/3x3_training_set.npz')
    valid_vit_dataset = LabelledPixels('data/3x3_test_set.npz')
    train_ae_dataset  = UnlabelledPixels('data/3x3_reflectance_1.npz', range(50000))
    valid_ae_dataset  = UnlabelledPixels('data/3x3_reflectance_1.npz', range(50000, 52000))
    
    trainer = Trainer(
        vit, ae,
        train_vit_dataset, valid_vit_dataset,
        train_ae_dataset, valid_ae_dataset,
        batch_size,
        device
    )
    trainer.run()
    
    return vit

def infer(vit, data_paths):
    for file in tqdm(data_paths):
        reflectance = get_reflectance(os.path.join('data', file))
        
        image = []
        for h in tqdm(range(reflectance.shape[0] - 3)):
            row = extract_patches_2d(reflectance[h:h + 3], (3, 3)).transpose(0, 3, 1, 2)
            prediction = vit(torch.tensor(row)).argmax(dim = 1).detach().numpy()
            image.append(prediction)
        image = numpy.stack(image)
        
        cv2.imwrite(f'rollout/unsupervised_vit/{file}.png', image * 255)
    
if __name__ == '__main__':
    DATA_PATHS = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
    
    vit = train()
    infer(vit, DATA_PATHS)