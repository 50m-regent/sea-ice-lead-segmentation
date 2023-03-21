import numpy
import os
from tqdm import tqdm
import cv2
from sklearn.ensemble import RandomForestClassifier

from utils import load_labelled_data, get_reflectance

def rollout(file, save_path, save_name, model):
    reflectance = get_reflectance(file)
    image = numpy.asarray([
        model.predict(row) * 255
        for row in reflectance
    ], dtype = numpy.uint8)
    
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    cv2.imwrite(
        os.path.join(save_path, f'{save_name}.png'),
        image
    )

def random_forest(data_size):
    train_data, (test_x, test_y) = load_labelled_data(data_size)
    
    classifier = RandomForestClassifier()
    classifier.fit(*train_data)
    
    prediction = classifier.predict(test_x)
    accuracy = numpy.count_nonzero(prediction == test_y) / len(test_y)
    print(f'Accuracy: {accuracy}')
    
    files = os.listdir('data/reflectance')
    for file in tqdm(files):
        rollout(
            os.path.join('data/reflectance', file),
            'rollout/random_forest',
            file,
            classifier
        )
    
if __name__ == '__main__':
    random_forest(1)