import os
import numpy
import cv2

from utils import get_reflectance, normalize

def detect_edge(file, save_path, save_name, lower_threshold, upper_threshold):
    reflectance = get_reflectance(file)[:, :, 0]
    reflectance = (normalize(reflectance) * 255).astype(numpy.uint8)
    
    edges = cv2.Canny(reflectance, lower_threshold, upper_threshold)
    
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    cv2.imwrite(
        os.path.join(save_path, f'{save_name}_{lower_threshold}_{upper_threshold}.png'),
        edges
    )
    
def main():
    files = os.listdir('data/reflectance')
    for file in files:
        for lower_threshold in range(0, 128, 16):
            for upper_threshold in range(lower_threshold + 16, 128, 16):
                detect_edge(
                    os.path.join('data/reflectance', file),
                    'rollout/edge_detection',
                    file,
                    lower_threshold, upper_threshold
                )

if __name__ == '__main__':
    main()