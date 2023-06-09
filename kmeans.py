import os
import numpy
from sklearn.cluster import KMeans
import cv2
from tqdm import tqdm

from utils import load_labelled_data, load_unlabelled_data, get_reflectance, patch_image

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
            
def patched_kmeans(n_clusters, patch_size):
    files = os.listdir('data/reflectance')
    for file in tqdm(files):
        reflectance = get_reflectance(os.path.join('data/reflectance', file))
        patches     = patch_image(reflectance, patch_size)
        
        image = []
        for row in patches:
            image_row = []
            for patch in row:
                classifier = KMeans(n_clusters, n_init = 'auto')
                prediction = classifier.fit_predict(
                    patch.reshape(-1, patch.shape[-1])
                ).reshape(patch.shape[:2])
                
                darkest_cluster = numpy.argsort(classifier.cluster_centers_, axis = 0)[:, 0][0]
                
                patch = numpy.where(prediction == darkest_cluster, 255, 0)
                image_row.append(patch)
                
            image.append(numpy.concatenate(image_row, axis = 1))
            
        image = numpy.concatenate(image)
        cv2.imwrite(f'rollout/kmeans/{file}_{n_clusters}_{patch_size}_patched.png', image)
            
def kmeans_main(data_size):
    n_clusters = 8
    
    train_data, test_data = load_labelled_data(data_size)
    unlabelled_x = load_unlabelled_data(data_size)
    train_x      = numpy.concatenate((unlabelled_x, train_data[0]))
    
    kmeans(train_x, test_data, n_clusters)
            
if __name__ == '__main__':
    n_clusters = 2
    patch_size = 64
    
    patched_kmeans(n_clusters, patch_size)