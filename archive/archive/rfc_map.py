import os
import cv2
import numpy

from utils.load_data import load_rfc_labelled_data, load_test_data
from utils.get_reflectance import get_reflectance
from rfc import random_forest

DATA_PATH = 'data'
IMAGE_PATHS = [
    '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'
]
SAVE_PATH = 'rfc_map'

def map_over(rfc, path):
    reflectance = get_reflectance(path)
    prediction  = rfc.predict(
        reflectance.reshape(-1, reflectance.shape[-1])
    ).reshape(reflectance.shape[:-1])
    
    return prediction

def map_over_3x3(rfc, path):
    reflectance = get_reflectance(path)
    data = []
    for h in range(reflectance.shape[0] - 3):
        for w in range(reflectance.shape[1] - 3):
            data.append(reflectance[h:h+3, w:w+3].reshape(-1,))
            
    data = numpy.array(data)
    
    prediction  = rfc.predict(
        data
    ).reshape((reflectance.shape[0] - 3, reflectance.shape[1] - 3))
    
    return prediction

if __name__ == '__main__':
    # train_x, train_y = load_rfc_labelled_data()
    train_x, train_y = load_test_data()
    rfc = random_forest(train_x, train_y)
    
    for file_name in IMAGE_PATHS:
        image = map_over(rfc, os.path.join(DATA_PATH, file_name))
        cv2.imwrite(
            os.path.join(SAVE_PATH, file_name + '.png'),
            image * 255
        )
        
        print(f'Saved: {os.path.join(SAVE_PATH, file_name + ".png")}')
    