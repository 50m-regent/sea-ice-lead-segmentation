import numpy

def normalize(x):
    return (x - numpy.min(x)) / (numpy.max(x) - numpy.min(x))