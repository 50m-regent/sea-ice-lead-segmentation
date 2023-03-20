from sklearn.cluster import KMeans
from matplotlib import pyplot
import numpy
import sys

from utils.readfiles import readfiles

DATA_PATH = 'data/S3B_OL_1_EFR____20190301T232521_20190301T232821_20200111T235148_0179_022_301_1800_MR1_R_NT_002.SEN3'

if __name__ == '__main__':
    n_clusters = 2
    if len(sys.argv) == 2:
        n_clusters = int(sys.argv[1])
    
    dataset = readfiles(DATA_PATH)
    
    all_data = numpy.stack(dataset[:, 1], axis=2)

    kmeans   = KMeans(n_clusters=n_clusters, n_init='auto').fit(all_data.reshape(-1, all_data.shape[2]))
    clusters = kmeans.labels_.reshape(all_data.shape[:2])
    cluster  = numpy.where(kmeans.labels_ == numpy.argmin(numpy.linalg.norm(kmeans.cluster_centers_, axis=1)), 0, 1).reshape(all_data.shape[:2])
    
    pyplot.imshow(cluster)
    pyplot.savefig(f'k-means/{n_clusters}/all.png', dpi=1200)
    
    pyplot.imshow(clusters)
    pyplot.savefig(f'k-means/{n_clusters}/all_bright.png', dpi=1200)
    