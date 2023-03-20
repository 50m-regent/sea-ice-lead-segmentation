from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import numpy

from utils.load_data import load_klustering_labelled_data, load_test_data

SEARCH_PARAMETERS = {
    'n_estimators'      : [5, 10, 20, 30, 50, 100, 300],
    'max_features'      : [3, 5, 10, 15, 20],
    'n_jobs'            : [-1],
    'min_samples_split' : [3, 5, 10, 15, 20, 25, 30, 40, 50, 100],
    'max_depth'         : [3, 5, 10, 15, 20, 25, 30, 40, 50, 100]
}

def grid_search(train_x, train_y):
    gs = GridSearchCV(
        RandomForestClassifier(),
        SEARCH_PARAMETERS,
        cv=3,
        verbose=True,
        n_jobs=-1
    )
    gs.fit(train_x, train_y)
    print(gs.best_estimator_)
    
    return gs.best_estimator_

def random_forest(train_x, train_y):
    rfc = RandomForestClassifier(verbose=True, n_jobs=-1)
    rfc.fit(train_x, train_y)
    
    return rfc

def calculate_accuracy(rfc, x, y):
    prediction = rfc.predict(x)
    accuracy   = numpy.count_nonzero(prediction == y) / y.shape[0]
    
    return accuracy

if __name__ == '__main__':
    train_x, train_y = load_klustering_labelled_data()
    test_x, test_y   = load_test_data()
    
    rfc = grid_search(train_x, train_y)
    # rfc = random_forest(train_x, train_y)
    
    accuracy = calculate_accuracy(rfc, test_x, test_y)
    print(f'Accuracy: {accuracy:>0}')
    