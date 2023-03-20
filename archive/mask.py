import os
import cv2
from tqdm import tqdm
import numpy

from utils.get_reflectance import get_reflectance
from utils.mask import inspect_by_blur, get_mask
from utils.normalize import normalize

DATA_PATH = 'data'
IMAGE_PATHS = [
    '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'
]

def mask_blur(file_name):
    reflectance = get_reflectance(os.path.join(DATA_PATH, file_name))[:, :, 0]
    blur_map    = get_mask(os.path.join(DATA_PATH, file_name), inspect_by_blur, region_size=128)
    
    reflectance = normalize(reflectance) * 196
    blur_map    = normalize(-blur_map) * 55
    
    image = numpy.stack((reflectance + blur_map, reflectance, reflectance), axis = 2)
    
    cv2.imwrite(f'rollout/blur_mask/{file_name}.png', image)
    cv2.imwrite(f'rollout/blur_mask/{file_name}_blur.png', blur_map * 4)

if __name__ == '__main__':
    for file_name in tqdm(IMAGE_PATHS):
        mask_blur(file_name)
    