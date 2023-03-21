import os
import cv2

from utils import get_reflectance, normalize

def visualize(file, save_path, save_name):
    reflectance = get_reflectance(file).transpose(2, 0, 1)
    
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    for c, channel in enumerate(reflectance):
        channel = normalize(channel) * 255
        cv2.imwrite(
            os.path.join(save_path, f'{save_name}_{c}.png'),
            channel
        )

def main():
    files = os.listdir('data/reflectance')
    for file in files:
        visualize(
            os.path.join('data/reflectance', file),
            'rollout/reflectance',
            file
        )

if __name__ == '__main__':
    main()