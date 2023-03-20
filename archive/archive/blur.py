import cv2
import sys
import numpy

from utils.readfiles import readfiles
from utils.normalize import normalize

DATA_PATH = 'data/S3B_OL_1_EFR____20190301T232521_20190301T232821_20200111T235148_0179_022_301_1800_MR1_R_NT_002.SEN3'

if __name__ == '__main__':
    k = 8
    if len(sys.argv) == 2:
        k = int(sys.argv[1])
    
    data = readfiles(DATA_PATH)
    mean_data = numpy.array(
        normalize(
            numpy.mean(numpy.stack(data[:, 1], axis=2), axis=2)
        ) * 255,
        dtype=numpy.uint8
    )
    
    blurred = cv2.blur(mean_data, (k, k))
    cv2.imwrite(f'blur/normal_{k}.png', blurred)
    
    for _ in range(10):
        blurred = cv2.blur(blurred, (k, k))
    cv2.imwrite(f'blur/twice_{k}.png', blurred)