import numpy
from sklearn.ensemble import RandomForestClassifier

from utils.load_data import load_test_data, load_train_data

def calculate_accuracy(model, x, y):
    prediction = model.predict(x)
    accuracy   = numpy.count_nonzero(prediction == y) / y.shape[0]
    
    return accuracy

def random_forest(x, y):
    rfc = RandomForestClassifier(n_jobs=-1, verbose=True)
    rfc.fit(x, y)
    
    return rfc

if __name__ == '__main__':
    '''
    train_x, train_y = load_3x3_labelled_data()
    test_x           = load_3x3_unlabelled_data()
    
    rfc = random_forest(train_x, train_y)
    
    accuracy = calculate_accuracy(rfc, train_x, train_y)
    print(f'Accuracy: {accuracy:>.7f}')
    
    labels = rfc.predict(test_x).reshape(-1,)
    
    print(test_x.shape)
    print(labels.shape)
    
    numpy.savez_compressed(
        f'data/rfc_labelled.npz',
        x=test_x.astype(numpy.float32),
        y=labels.astype(numpy.float32)
    )
    '''
    
    train_x, train_y = load_test_data()
    test_x           = load_train_data()[:1000000]
    
    rfc = random_forest(train_x, train_y)
    
    accuracy = calculate_accuracy(rfc, train_x, train_y)
    print(f'Accuracy: {accuracy:>.7f}')
    
    labels = rfc.predict(test_x).reshape(-1,)
    
    print(test_x.shape)
    print(labels.shape)
    
    numpy.savez_compressed(
        f'data/rfc_labelled.npz',
        x=test_x.astype(numpy.float32),
        y=labels.astype(numpy.float32)
    )
