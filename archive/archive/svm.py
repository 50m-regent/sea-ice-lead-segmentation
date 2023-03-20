import numpy
from sklearn.svm import OneClassSVM

from utils.load_data import load_train_data, load_test_data

if __name__ == '__main__':
    train_x        = load_train_data()
    test_x, test_y = load_test_data()
    
    numpy.random.shuffle(train_x)
    train_x = train_x[:100000]
    
    svm = OneClassSVM(
        gamma='auto',
        cache_size=5000,
        verbose=True
    ).fit(train_x)
    prediction = numpy.where(-1 == svm.predict(test_x), 1, 0)
    
    print(numpy.count_nonzero(test_y) / test_y.shape[0])
    
    print(f'Accuracy: {numpy.count_nonzero(prediction == test_y) / len(test_y)}')
    