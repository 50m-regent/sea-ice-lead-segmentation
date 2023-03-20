from pyclustering.cluster.kmedians import kmedians
from matplotlib import pyplot
import numpy
import sys

from utils.readfiles import readfiles

DATA_PATH = 'data/S3B_OL_1_EFR____20190301T232521_20190301T232821_20200111T235148_0179_022_301_1800_MR1_R_NT_002.SEN3'

if __name__ == '__main__':
    n_clusters = 2
    #if len(sys.argv) == 2:
    #    n_clusters = int(sys.argv[1])
    
    dataset = readfiles(DATA_PATH)
    
    all_data = numpy.stack(dataset[:, 1], axis=2)
    most_bright = numpy.max(all_data, axis=2)

    kmed = kmedians(most_bright.reshape(-1, 1), [numpy.min(most_bright), numpy.max(most_bright)])
    kmed.process()
    clusters = kmed.get_clusters()
    
    pyplot.imshow(clusters)
    pyplot.savefig(f'k-medians/{n_clusters}.png', dpi=1200)
    