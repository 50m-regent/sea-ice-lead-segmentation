from sklearn.cluster import KMeans
from matplotlib import pyplot
import numpy
import sys
import cv2
import os

from utils.get_reflectance import get_reflectance
from utils.normalize import normalize

DATA_PATH = 'data'
FILE_PATHS = [
    '1',
    '2',
    '3',
    '4',
    '5',
    '6',
    '7',
    '8',
    '9',
    '10',
    '11',
    '12'
]

if __name__ == '__main__':
    n_clusters = 2
    if len(sys.argv) == 2:
        n_clusters = int(sys.argv[1])
        
    for file_path in FILE_PATHS:
        data = get_reflectance(
            os.path.join(DATA_PATH, file_path)
        )

        kmeans   = KMeans(n_clusters=n_clusters, n_init='auto').fit(data.reshape(-1, data.shape[2]))
        clusters = normalize(kmeans.labels_.reshape(data.shape[:2])) * 255
        cv2.imwrite(f'k-means_reflectance/{file_path}/{n_clusters}/all.png', clusters)
        
        for i in range(1, n_clusters + 1):
            args = list(numpy.argsort(kmeans.cluster_centers_, axis=0)[:, 0][:i])
            clusters = numpy.where(
                [element in args for element in kmeans.labels_], 0, 255
            ).reshape(
                data.shape[:2]
            )
            cv2.imwrite(f'k-means_reflectance/{file_path}/{n_clusters}/{i}.png', clusters)
            
            print(f'Saved: {i}/{n_clusters}')
    