import os
import numpy

from utils.get_reflectance import get_reflectance

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
        data_3x3s = []
        for h in range(data.shape[0] - 3):
            for w in range(data.shape[1] - 3):
                data_3x3s.append(data[h:h + 3, w:w + 3])
                
        data_3x3s = numpy.array(data_3x3s)[:100000]
        print(data_3x3s.shape)
            
        numpy.savez_compressed(
            f'data/3x3_reflectance_{file_name}.npz',
            data_3x3s.astype(numpy.float32)
        )
        
        del data_3x3s
    