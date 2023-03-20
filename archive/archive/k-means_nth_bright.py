from sklearn.cluster import KMeans
from matplotlib import pyplot
import numpy
import sys

from utils.readfiles import readfiles

DATA_PATH  = 'data/S3B_OL_1_EFR____20190301T232521_20190301T232821_20200111T235148_0179_022_301_1800_MR1_R_NT_002.SEN3'
N_CLUSTERS = 20

if __name__ == '__main__':
    n_clusters = 2
    if len(sys.argv) == 2:
        n_clusters = int(sys.argv[1])
    
    dataset = readfiles(DATA_PATH)
    
    all_data = numpy.stack(dataset[:, 1], axis=2)
    most_bright = numpy.mean(all_data, axis=2)

    kmeans   = KMeans(n_clusters=N_CLUSTERS, n_init='auto').fit(most_bright.reshape(-1, 1))
    
    for n in range(1, n_clusters + 1):
        args = list(numpy.argsort(kmeans.cluster_centers_.flatten())[:n])
        clusters = numpy.where(
            [element in args for element in kmeans.labels_], 0, 1
        ).reshape(
            most_bright.shape
        )
        
        pyplot.imshow(clusters)
        pyplot.savefig(f'k-means/{n_clusters}/{n}th_bright.png', dpi=1200)
    
        print(f"Plotted: {n}/{n_clusters}")