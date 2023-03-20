import cv2
import os
import numpy

from utils.get_reflectance import get_reflectance
from utils.normalize import normalize

DATA_PATH = 'data'
FILE_NAMES = [
    '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'
]

if __name__ == '__main__':
    all_reflectance = None
    
    for file_name in FILE_NAMES:
        data = get_reflectance(
            os.path.join(DATA_PATH, file_name)
        )
        
        if None is all_reflectance:
            all_reflectance = data.reshape(-1, data.shape[2])
        else:
            all_reflectance = numpy.concatenate((all_reflectance, data.reshape(-1, data.shape[2])))

        if not os.path.exists(f'reflectance/{file_name}'):
            os.makedirs(f'reflectance/{file_name}')
            
        for i in range(data.shape[2]):
            cv2.imwrite(f'reflectance/{file_name}/{i + 1}.png', normalize(data[:, :, i]) * 255)
            print(f'Saved: {file_name} {i + 1}/{data.shape[2]} {data[:, :, i].shape[0]}x{data[:, :, i].shape[1]}')
            
    print(all_reflectance.shape)
    numpy.savez_compressed('data/all_reflectance.npz', all_reflectance.astype(numpy.float32))
    print('Saved')
