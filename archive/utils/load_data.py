import numpy
import os
from tqdm import tqdm
import random

def load_reflectance_358():
    with numpy.load('data/reflectance_358.npz') as f:
        x = f['arr_0'].astype(numpy.float32)
        numpy.random.shuffle(x)
        return x
    
def load_labelled_358():
    with numpy.load('data/labelled_358.npz') as f:
        x = f['x'].astype(numpy.float32)
        y = f['y'].astype(numpy.float32)
        
        y = y[numpy.where((x < 100).all(axis=1))[0]]
        x = x[numpy.where((x < 100).all(axis=1))[0]]
        
    assert len(x) == len(y)
    permutation = numpy.random.permutation(len(x))
    
    return x[permutation], y[permutation]
        

def load_train_data():
    with numpy.load('data/all_reflectance.npz') as f:
        x = f['arr_0'].astype(numpy.float32)
        numpy.random.shuffle(x)
        return x

def load_test_data():
    with numpy.load('data/1x1_test_set.npz') as f:
        x = f['x'].astype(numpy.float32)
        y = f['y'].astype(numpy.float32)
        
        y1 = y[numpy.where((x < 100).all(axis=1))[0]]
        x1 = x[numpy.where((x < 100).all(axis=1))[0]]
        
    with numpy.load('data/1x1_training_set.npz') as f:
        x = f['x'].astype(numpy.float32)
        y = f['y'].astype(numpy.float32)
        
        y2 = y[numpy.where((x < 100).all(axis=1))[0]]
        x2 = x[numpy.where((x < 100).all(axis=1))[0]]
        
    x = numpy.concatenate([x1, x2])
    y = numpy.concatenate([y1, y2])
    
    assert len(x) == len(y)
    permutation = numpy.random.permutation(len(x))
        
    return x[permutation], y[permutation]

def load_3x3_unlabelled_data():
    FILE_NAMES = [
        '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'
    ]
    
    data = None
    for file_name in FILE_NAMES:
        with numpy.load(f'data/3x3_reflectance_{file_name}.npz') as f:
            x = f['arr_0'].astype(numpy.float32)
            numpy.random.shuffle(x)

            if None is data:
                data = x
            else:
                data = numpy.concatenate((data, x))
                
            del x
            
            print(f'Loaded: {file_name}')
                
    data = numpy.array([
        datum.reshape(3 * 3 * data.shape[-1],) for datum in data
    ])
    numpy.random.shuffle(data)
    return data

def load_3x3_labelled_data():
    with numpy.load('data/3x3_training_set.npz') as f:
        x1 = f['x'].astype(numpy.float32)
        y1 = f['y'].astype(numpy.float32)
        
    with numpy.load('data/3x3_test_set.npz') as f:
        x2 = f['x'].astype(numpy.float32)
        y2 = f['y'].astype(numpy.float32)
        
    x = numpy.concatenate([x1, x2])
    y = numpy.concatenate([y1, y2])
    
    x = numpy.array([xx.reshape(3 * 3 * x.shape[-1],) for xx in x])
    
    assert len(x) == len(y)
    permutation = numpy.random.permutation(len(x))
    
    return x[permutation], y[permutation]

def load_rfc_labelled_data():
    with numpy.load('data/rfc_labelled.npz') as f:
        x = f['x'].astype(numpy.float32)
        y = f['y'].astype(numpy.float32)
        
    assert len(x) == len(y)
    permutation = numpy.random.permutation(len(x))
        
    return x[permutation], y[permutation]

def load_klustering_labelled_data():
    with numpy.load('data/klustering_labelled.npz') as f:
        x = f['x'].astype(numpy.float32)
        y = f['y'].astype(numpy.float32)
        
    assert len(x) == len(y)
    permutation = numpy.random.permutation(len(x))
        
    return x[permutation], y[permutation]

def load_trimmed_file(file_name):
    with numpy.load(os.path.join('data/trimmed', file_name)) as f:
        return f['x'].astype(numpy.float32)
    
def load_trimmed(n = 10000):
    files = os.listdir('data/trimmed')
    files = [file for file in files if 'npz' in file]
    random.shuffle(files)
    files = files[:n]
    
    data = [load_trimmed_file(file) for file in tqdm(files, desc = 'Loading')]
    return numpy.asarray(data)

if __name__ == '__main__':
    x, y = load_rfc_labelled_data()
    
    print(x.shape)
    print(y.shape)
    
    print(x)
    print(y)
