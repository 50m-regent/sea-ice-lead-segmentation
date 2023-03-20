from matplotlib import pyplot
import numpy
import pywt

from utils.readfiles import readfiles

DATA_PATH = 'data/S3B_OL_1_EFR____20190301T232521_20190301T232821_20200111T235148_0179_022_301_1800_MR1_R_NT_002.SEN3'

if __name__ == '__main__':
    dataset = readfiles(DATA_PATH)
    
    mean_data = numpy.mean(numpy.stack(dataset[:, 1], axis=2), axis=2)
    pyplot.imshow(mean_data)
    pyplot.savefig(f'wavelet/raw.png', dpi=1200)
    
    wavelet   = pywt.dwt2(mean_data, 'bior1.1')
    approximation, (horizontal_edge, verticle_edge, diagonal_edge) = wavelet
    
    pyplot.imshow(approximation)
    pyplot.savefig(f'wavelet/approximation.png', dpi=1200)
    
    pyplot.imshow(horizontal_edge)
    pyplot.savefig(f'wavelet/horizontal_edge.png', dpi=1200)
    
    pyplot.imshow(verticle_edge)
    pyplot.savefig(f'wavelet/verticle_edge.png', dpi=1200)
    
    pyplot.imshow(diagonal_edge)
    pyplot.savefig(f'wavelet/diagonal_edge.png', dpi=1200)
    