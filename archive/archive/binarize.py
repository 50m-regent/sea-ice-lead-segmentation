from sklearn.cluster import KMeans
from matplotlib import pyplot
import numpy

from utils.readfiles import readfiles
from utils.normalize import normalize

DATA_PATH = 'data/S3B_OL_1_EFR____20190301T232521_20190301T232821_20200111T235148_0179_022_301_1800_MR1_R_NT_002.SEN3'
# DATA_PATH = 'data/S3A_OL_1_EFR____20180307T054004_20180307T054119_20180308T091959_0075_028_319_1620_LN1_O_NT_002.SEN3'
THRESHOLD = 0.075



if __name__ == '__main__':
    dataset = readfiles(DATA_PATH)
    
    all_data = numpy.stack(dataset[:, 1], axis=2)
    most_bright = normalize(numpy.max(all_data, axis=2))
    

    clusters = numpy.where(most_bright < THRESHOLD, 0, 1).reshape(most_bright.shape)
    
    pyplot.imshow(clusters)
    pyplot.savefig(f'binarized{THRESHOLD}_2.png', dpi=1200)
    