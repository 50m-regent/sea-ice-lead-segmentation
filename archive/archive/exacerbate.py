from sklearn.cluster import KMeans
from matplotlib import pyplot
import numpy
import sys

from utils.readfiles import readfiles
from utils.normalize import normalize

DATA_PATH = 'data/S3B_OL_1_EFR____20190301T232521_20190301T232821_20200111T235148_0179_022_301_1800_MR1_R_NT_002.SEN3'

if __name__ == '__main__':
    exacerbation = 2
    n_clusters = 8
    if len(sys.argv) == 2:
        n_clusters = int(sys.argv[1])
    
    dataset = readfiles(DATA_PATH)
    
    data = normalize(numpy.max(numpy.stack(dataset[:, 1], axis=2), axis=2))
    data = numpy.where(data < 0.3, data, 0.3) ** exacerbation
    
    kmeans   = KMeans(n_clusters=n_clusters, n_init='auto').fit(data.reshape(-1, 1))
    clusters = numpy.where(kmeans.labels_ == numpy.argmin(kmeans.cluster_centers_), 0, 1).reshape(data.shape)
    
    pyplot.imshow(clusters)
    pyplot.savefig(f'exacerbation/exacerbate.png', dpi=1200)
    