import os
import numpy
from sklearn.cluster import KMeans
import cv2
from tqdm import tqdm

from utils import load_labelled_data, load_unlabelled_data, get_reflectance

def rollout(file, save_path, save_name, kmeans, n):
    lead_clusters = list(numpy.argsort(kmeans.cluster_centers_, axis = 0)[:, 0][:n])
    
    reflectance = get_reflectance(file)
    image = numpy.asarray([
        numpy.where(
            numpy.isin(kmeans.predict(row), lead_clusters),
            255, 0
        )
        for row in reflectance
    ], dtype = numpy.uint8)
    
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    cv2.imwrite(
        os.path.join(save_path, f'{save_name}_{n}.png'),
        image
    )
    tqdm.write(
        f'Saved: {os.path.join(save_path, f"{save_name}_{n}.png")}'
    )

def kmeans(train_data, test_data, n_clusters):
    classifier = KMeans(n_clusters, n_init = 'auto')
    classifier.fit(train_data)
    
    test_x, test_y = test_data
    
    files = os.listdir('data/reflectance')
    for file in tqdm(files):
        for n in range(1, n_clusters):
            lead_clusters = list(numpy.argsort(classifier.cluster_centers_, axis = 0)[:, 0][:n])
            
            prediction = classifier.predict(test_x)
            accuracy   = numpy.count_nonzero(numpy.isin(prediction, lead_clusters) == test_y) / len(test_y)
            tqdm.write(f'Accuracy: {accuracy}')
            
            rollout(
                os.path.join('data/reflectance', file),
                'rollout/kmeans',
                file,
                classifier, n
            )
            
if __name__ == '__main__':
    n_clusters = 8
    
    train_data, test_data = load_labelled_data(1)
    unlabelled_x = load_unlabelled_data(1)
    train_x      = numpy.concatenate((unlabelled_x, train_data[0]))
    
    kmeans(train_x, test_data, n_clusters)