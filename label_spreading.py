import numpy
import os
from tqdm import tqdm
import cv2
from sklearn.semi_supervised import LabelSpreading

from utils import load_labelled_data, load_unlabelled_data, get_reflectance

def rollout(file, save_path, save_name, model):
    reflectance = get_reflectance(file)
    image = numpy.asarray([
        model.predict(row) * 255
        for row in tqdm(reflectance)
    ], dtype = numpy.uint8)
    
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    cv2.imwrite(
        os.path.join(save_path, f'{save_name}.png'),
        image
    )

def label_spreading(data_size):
    (labelled_train_x, labelled_train_y), (test_x, test_y) = load_labelled_data(data_size)
    unlabelled_train_x = load_unlabelled_data(data_size)[:20000]
    unlabelled_train_y = numpy.full(len(unlabelled_train_x), -1)
    
    train_x = numpy.concatenate((labelled_train_x, unlabelled_train_x))
    train_y = numpy.concatenate((labelled_train_y, unlabelled_train_y))
    
    classifier = LabelSpreading()
    classifier.fit(train_x, train_y)
    
    prediction = classifier.predict(test_x)
    accuracy = numpy.count_nonzero(prediction == test_y) / len(test_y)
    print(f'Accuracy: {accuracy}')
    
    files = os.listdir('data/reflectance')
    for file in tqdm(files):
        rollout(
            os.path.join('data/reflectance', file),
            'rollout/label_spreading',
            file,
            classifier
        )
    
if __name__ == '__main__':
    label_spreading(1)