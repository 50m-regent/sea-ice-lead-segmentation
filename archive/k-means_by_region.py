from tqdm import tqdm
import os
import numpy
from sklearn.cluster import KMeans
import cv2

from utils.get_reflectance import get_reflectance
from utils.mask import get_mask, inspect_by_difference

DATA_PATH = 'data'
IMAGE_PATHS = [
    '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'
]

def k_means(image, n_clusters = 8):
    classifier = KMeans(n_clusters = n_clusters, n_init = 'auto')
    classifier.fit(image.reshape(-1, image.shape[-1]))
    
    args = list(numpy.argsort(classifier.cluster_centers_, axis = 0)[:, 0][:2])
    prediction = numpy.where(
        [label in args for label in classifier.labels_], 255, 0
    ).reshape(image.shape[:2])
    
    return prediction

def k_means_by_region(file_name, region_size = 64):
    reflectance = get_reflectance(os.path.join(DATA_PATH, file_name))
    height, width, _ = reflectance.shape
    
    prediction = numpy.zeros((height, width))
    
    for h in range(0, height, region_size):
        for w in range(0, width, region_size):
            trimmed = reflectance[h:h + region_size, w:w + region_size]
            
            prediction[h:h + region_size, w:w + region_size] = k_means(trimmed)
            
    return prediction

if __name__ == '__main__':
    for file_name in tqdm(IMAGE_PATHS):
        image  = k_means_by_region(file_name)
        masked = numpy.multiply(image, get_mask(os.path.join(DATA_PATH, file_name), inspect_by_difference))
        
        cv2.imwrite(f'rollout/k-means_by_region/{file_name}.png', image)
        cv2.imwrite(f'rollout/k-means_by_region/{file_name}_masked.png', masked)