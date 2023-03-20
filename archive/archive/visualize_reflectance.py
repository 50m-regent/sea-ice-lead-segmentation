import cv2
import os

from utils.get_reflectance import get_reflectance
from utils.normalize import normalize

DATA_PATH   = 'data'
IMAGE_PATHS = [
    '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'
]

if __name__ == '__main__':
    for file_name in IMAGE_PATHS:
        data = get_reflectance(
            os.path.join(DATA_PATH, file_name)
        )
        
        for i in range(data.shape[2]):
            # cv2.imwrite(f'reflectance/{file_name}/{i + 1}.png', normalize(data[:, :, i]) * 255)
            cv2.imwrite(f'reflectance/{file_name}/{i + 1}_test.png', normalize(data[:, :, i]) * 255)
            print(f"Saved: {i + 1}/{data.shape[2]} {data[:, :, i].shape[0]}x{data[:, :, i].shape[1]}")
