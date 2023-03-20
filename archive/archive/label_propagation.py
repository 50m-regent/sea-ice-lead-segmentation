import numpy
from sklearn.semi_supervised import LabelPropagation

from utils.load_data import load_train_data, load_test_data
from utils.split_data import split_data

if __name__ == '__main__':
    train_ratio =  0.5
    
    unlabelled_x = load_train_data()[:10000]
    unlabelled_y = numpy.full(len(unlabelled_x), -1)
    
    labelled_x, labelled_y = load_test_data()
    labelled_x, test_x     = split_data(labelled_x, train_ratio)
    labelled_y, test_y     = split_data(labelled_y, train_ratio)
    
    train_x = numpy.concatenate((unlabelled_x, labelled_x))
    train_y = numpy.concatenate((unlabelled_y, labelled_y))
    
    print(train_x.shape)
    print(test_x.shape)
    
    lp = LabelPropagation(n_jobs=-1)
    lp.fit(train_x, train_y)
    
    prediction = lp.predict(test_x)
    
    print(prediction)
    print(test_y)
    
    print(numpy.count_nonzero(prediction == test_y) / len(test_y))
    