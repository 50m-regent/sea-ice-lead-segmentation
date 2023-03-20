import os
import numpy
import cv2

from utils.get_reflectance import get_reflectance

DATA_PATH = 'data'
IMAGE_PATHS = [
    '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'
]
SAVE_PATH = 'rollout/edge_detection'

def detect_edge(file_name, lower_threshold, upper_threshold):
    reflectance = get_reflectance(os.path.join(DATA_PATH, file_name))[:, :, 0]
    reflectance = (reflectance * 255).astype(numpy.uint8)
    
    edge_image = cv2.Canny(reflectance, lower_threshold, upper_threshold)#, apertureSize=3)
    
    cv2.imwrite(
        os.path.join(SAVE_PATH, file_name + f'_{lower_threshold}_{upper_threshold}.png'),
        edge_image
    )
    print(f'Saved: {os.path.join(SAVE_PATH, file_name + f"_{lower_threshold}_{upper_threshold}.png")} {edge_image.shape}')

if __name__ == '__main__':
    for file_name in IMAGE_PATHS:
        for u in range(16, 256, 16):
            detect_edge(file_name, 0, u)
    