import os
import numpy
import cv2

from utils import get_reflectance, normalize, patch

def detect_blur(file, save_path, save_name, patch_size):
    reflectance = get_reflectance(file)[:, :, 0]
    reflectance = normalize(reflectance)
    
    patches = patch(reflectance, patch_size)
    
    image = [
        [cv2.Laplacian(patch, cv2.CV_32F).var() for patch in row]
        for row in patches
    ]
    image = numpy.asarray(image, dtype = numpy.float32)
    image = normalize(image) * 255
    
    if not os.path.exists(save_path):
        os.mkdir(save_path)
        
    cv2.imwrite(
        os.path.join(save_path, f'{save_name}_{patch_size}.png'),
        image
    )

def main():
    files = os.listdir('data/reflectance')
    for file in files:
        for patch_size in (64, 128, 256, 512):
            detect_blur(
                os.path.join('data/reflectance', file),
                'rollout/blur_detection',
                file,
                patch_size
            )

if __name__ == '__main__':
    main()