import numpy

from utils.load_data import load_train_data, load_test_data

if __name__ == '__main__':
    # data = load_train_data()
    # data = data[:, [3, 5, 8]]
    
    # numpy.savez_compressed('data/reflectance_358.npz', data)
    
    data, y = load_test_data()
    data = data[:, [3, 5, 8]]
    
    numpy.savez_compressed('data/labelled_358.npz', x=data, y=y)
    