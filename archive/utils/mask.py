import numpy
import cv2

from .get_reflectance import get_reflectance

def inspect_by_difference(image, threshold = 0.1):
    image = image.reshape(-1, image.shape[-1])
    
    max_distance = numpy.max(image) - numpy.min(image)
            
    return threshold < max_distance

def inspect_by_blur(image, threshold = 0.0005):
    score = cv2.Laplacian(image, cv2.CV_32F).var()
    
    return score

def get_mask(image_dir, inspector, region_size = 32):
    reflectance = get_reflectance(image_dir)[:, :, 0]
    mask        = numpy.zeros(reflectance.shape)

    height, width = reflectance.shape
    for h in range(0, height, region_size):
        for w in range(0, width, region_size):
            trimmed = reflectance[h:h + region_size, w:w + region_size]
            
            has_lead = inspector(trimmed)
            mask[h:h + region_size, w:w + region_size] = has_lead
    
    return mask
