import numpy
import torch
from torch.utils.data import Dataset

class UnlabelledPixels(Dataset):
    def __init__(self, path, i):
        with numpy.load(path) as f:
            self.data = f['x'].astype(numpy.float32)[i].transpose(0, 3, 1, 2)
            
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
    
class Patches(Dataset):
    def __init__(self, files):
        self.images = []
        
        for file in files:
            with numpy.load(file) as f:
                image1 = torch.tensor(f['x'], dtype = torch.float) / 255
            image2 = image1[numpy.random.permutation(image1.shape[0])]
                
            self.images.append(
                (image1, image2)
            )
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, i):
        return self.images[i]