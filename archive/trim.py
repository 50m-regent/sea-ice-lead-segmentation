import os
import numpy
import cv2
from tqdm import tqdm

from utils.get_reflectance import get_reflectance
from utils.normalize import normalize

DATA_PATH  = 'data'
IMAGE_DIRS = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
SAVE_DIR   = 'trimmed'

i = 0

def trim_image(file_path, save_path, size = 128, stride = 64):
    global i

    reflectance = get_reflectance(file_path)

    height, width, _ = reflectance.shape
    for x in range(0, width - size, stride):
        for y in range(0, height - size, stride):
            trimmed = reflectance[y:y + size, x:x + size].transpose(2, 0, 1)
            
            image = numpy.zeros(trimmed.shape, dtype = numpy.uint8)
            for c, channel in enumerate(trimmed):
                image[c] = normalize(channel) * 255

            i += 1

            numpy.savez_compressed(os.path.join(save_path, f'{i:>06}.npz'), x=image)
            tqdm.write(f"Saved: {os.path.join(save_path, f'{i:>06}')} {image.shape}")
            
            if 0 == i % 100:
                for c in range(21):
                    cv2.imwrite(os.path.join(save_path, f'{i:>06}_{c}.png'), image[c])
            
def save_all():
    for file_name in tqdm(IMAGE_DIRS, desc='Trimming'):
        file_path = os.path.join(DATA_PATH, file_name)
        save_path = os.path.join(DATA_PATH, SAVE_DIR)

        trim_image(file_path, save_path)

if __name__ == '__main__':
    save_all()
    
    '''
    size = 256
    stride = 64
    
    for file_name in tqdm(IMAGE_DIRS):
        file_path = os.path.join(DATA_PATH, file_name)
        save_path = os.path.join(DATA_PATH, SAVE_DIR)
        
        reflectance = get_reflectance(file_path)
        height, width, _ = reflectance.shape
        
        data = []
        for x in range(0, width - size, stride):
            for y in range(0, height - size, stride):
                trimmed = reflectance[y:y + size, x:x + size]
                
                data.append(trimmed)
                
        data = numpy.array(data, dtype = numpy.float32)

        numpy.savez_compressed(os.path.join(f'data/trimmed256_{file_name}.npz'), data)
        '''