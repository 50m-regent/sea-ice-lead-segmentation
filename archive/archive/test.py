import numpy
from sklearn.cluster import KMeans
import sys
import os
import tqdm

from utils.get_reflectance import get_reflectance

DATA_PATH = 'data'
TRAIN_PATH = [
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
TEST_PATH  = '1x1_test_set.npz'

if __name__ == '__main__':
    n_clusters = 2
    if len(sys.argv) == 2:
        n_clusters = int(sys.argv[1])
        
    kmeans = KMeans(n_clusters, n_init='auto')
    for train_file in tqdm.tqdm(TRAIN_PATH):
        train_data = get_reflectance(
            os.path.join(DATA_PATH, train_file)
        )
        kmeans.fit(train_data.reshape(-1, train_data.shape[2]))
    
    with numpy.load(os.path.join(DATA_PATH, TEST_PATH)) as test_data:
        x = test_data['x']
        y = test_data['y']
        
    for i in range(1, n_clusters + 1):
        args = list(numpy.argsort(kmeans.cluster_centers_, axis=0)[:, 0][:i])
        clusters = numpy.where(
            [element in args for element in kmeans.predict(x)], 1, 0
        )
        accuracy = len(numpy.where(clusters == y)[0]) / len(y)
        print(f'Accuracy {i}/{n_clusters}: {accuracy}')
