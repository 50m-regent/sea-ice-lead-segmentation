import cv2
import sys

IMAGE_PATH = 'k-means_reflectance/20/3.png'

if __name__ == '__main__':
    k = 3
    if len(sys.argv) == 2:
        k = int(sys.argv[1]) * 2 + 1
    
    data = cv2.imread(IMAGE_PATH)
    
    blurred = cv2.medianBlur(data, k)
    cv2.imwrite(f'median_filter/{k}.png', blurred)
    