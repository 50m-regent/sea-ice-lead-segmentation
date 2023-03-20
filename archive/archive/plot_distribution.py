import numpy
from matplotlib import pyplot

from utils.readfiles import readfiles

DATA_PATH = 'data/S3B_OL_1_EFR____20190301T232521_20190301T232821_20200111T235148_0179_022_301_1800_MR1_R_NT_002.SEN3'
# DATA_PATH = 'data/S3A_OL_1_EFR____20180307T054004_20180307T054119_20180308T091959_0075_028_319_1620_LN1_O_NT_002.SEN3'

if __name__ == '__main__':
    dataset = readfiles(DATA_PATH)
    
    all_data = numpy.stack(dataset[:, 1], axis=2)
    most_bright = numpy.max(all_data, axis=2).flatten()
    
    pyplot.scatter(numpy.linspace(0, 1, len(most_bright)), sorted(most_bright))
    pyplot.savefig('distribution.png', dpi=1200)
    