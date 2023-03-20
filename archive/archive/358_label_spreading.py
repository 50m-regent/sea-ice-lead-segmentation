import numpy
import os
import cv2
from sklearn.semi_supervised import LabelSpreading
from tqdm import tqdm

from utils.load_data import load_reflectance_358, load_labelled_358
from utils.split_data import split_data
from utils.get_reflectance import get_reflectance_358

numpy.seterr(invalid='ignore')

DATA_PATH = 'data'
IMAGE_PATHS = [
    '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'
]
SAVE_PATH = 'label_spreading_map'

def label_spreading(train_x, train_y, alpha=0.1):
    ls = LabelSpreading(kernel='knn', alpha=alpha, n_jobs=-1)
    ls.fit(train_x, train_y)
    
    return ls

def get_data(train_ratio = 0.6):
    unlabelled_x           = load_reflectance_358()
    unlabelled_y           = numpy.full(len(unlabelled_x), -1)
    labelled_x, labelled_y = load_labelled_358()
    
    labelled_x, test_x = split_data(labelled_x, train_ratio)
    labelled_y, test_y = split_data(labelled_y, train_ratio)
    
    train_x = numpy.concatenate((unlabelled_x[:50000], labelled_x))
    train_y = numpy.concatenate((unlabelled_y[:50000], labelled_y))
    
    return (train_x, train_y), (test_x, test_y)

def get_accuracy(ls, test_x, test_y):
    prediction = ls.predict(test_x)
    
    return numpy.count_nonzero(prediction == test_y) / len(test_y)

def rollout(file_name, ls):
    reflectance = get_reflectance_358(os.path.join(DATA_PATH, file_name))
    
    image = numpy.array([
        ls.predict(row) for row in tqdm(reflectance)
    ])
    
    cv2.imwrite(os.path.join(SAVE_PATH, f'{file_name}_358.png'), image * 255)
    print(f'Saved: {os.path.join(SAVE_PATH, f"{file_name}_358.png")}')

if __name__ == '__main__':
    (train_x, train_y), (test_x, test_y) = get_data()
    
    ls = label_spreading(train_x, train_y)

    accuracy = get_accuracy(ls, test_x, test_y)
    print(f'Accuracy: {accuracy}')
    
    for file_name in IMAGE_PATHS:
        rollout(file_name, ls)
