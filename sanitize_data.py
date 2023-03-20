import os
import numpy
from tqdm import tqdm
from sklearn.feature_extraction.image import extract_patches_2d

from utils import get_reflectance

def extract_pixels(size):
    DATA_PATH = 'data/reflectance'
    N_DATA    = 100000
    
    files = os.listdir(DATA_PATH)
    data = []
    for file in tqdm(files, desc = 'Loading'):
        reflectance = get_reflectance(os.path.join(DATA_PATH, file))
        
        datum = extract_patches_2d(reflectance, (size, size))
        numpy.random.shuffle(datum)
        datum = datum[:N_DATA]
        data.append(datum.squeeze())
    
    data = numpy.concatenate(data)
    numpy.random.shuffle(data)
    data = data[:N_DATA].transpose(0, 3, 1, 2)
    
    numpy.savez_compressed(f'data/{size}x{size}_unlabelled_set.npz', x=data)
    print(f'Saved: data/{size}x{size}_unlabelled_set.npz {data.shape}')
    
def _trim(image, size):
    image = image[:image.shape[0] // size * size, : image.shape[1] // size * size]
        
    image = numpy.split(image, range(size, image.shape[0], size))
    image = numpy.stack(image)

    image = numpy.split(image, range(size, image.shape[2], size), axis = 2)
    image = numpy.stack(image)
    
    image = image.reshape((-1, *image.shape[2:]))
    
    return image
    
def trim_images(size):
    DATA_PATH = 'data/reflectance'
    
    files = os.listdir(DATA_PATH)
    for file in tqdm(files, desc = 'Loading'):
        data = []
        
        reflectance = get_reflectance(os.path.join(DATA_PATH, file))
        data.append(_trim(reflectance, size))
        data.append(_trim(reflectance[size // 2:], size))
        data.append(_trim(reflectance[:, size // 2:], size))
        data.append(_trim(reflectance[size // 2:, size // 2:], size))
        
        data = numpy.concatenate(data)
        numpy.savez_compressed(f'data/{size}x{size}_unlabelled_set_{file}.npz', x=data)
        print(f'Saved: data/{size}x{size}_unlabelled_set_{file}.npz {data.shape}')
    
if __name__ == '__main__':
    # extract_pixels(1)
    # extract_pixels(3)
    
    # trim_images(size = 128)
    
    ...