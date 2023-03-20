import os
import numpy
import cv2

from utils.get_reflectance import get_reflectance
from utils.normalize import normalize

DATA_PATH = 'data'
IMAGE_PATHS = [
    '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'
]
SAVE_PATH = 'blur_detection'

def detect_blur(file_name, k = 5):
    reflectance = get_reflectance(os.path.join(DATA_PATH, file_name))[:, :, 0]
    reflectance = (reflectance * 255).astype(numpy.uint8)
    
    return cv2.Laplacian(reflectance, cv2.CV_64F, ksize = k)

if __name__ == '__main__':
    for file_name in IMAGE_PATHS:
        blur_map    = normalize(detect_blur(file_name, 3))
        reflectance = normalize(get_reflectance(os.path.join(DATA_PATH, file_name))[:, :, 0])
        
        blur_removed = normalize(reflectance - blur_map) * 255
        
        cv2.imwrite(f'blur_detection/masked/{file_name}.png', blur_removed)
