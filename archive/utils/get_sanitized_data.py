from .load_data import load_3x3_labelled_data
from .split_data import split_data

def get_sanitized_labelled_data(test_ratio = 0.8):
    train_x, train_y = load_3x3_labelled_data()
    train_x, test_x  = split_data(train_x, test_ratio)
    train_y, test_y  = split_data(train_y, test_ratio)
    
    return (train_x, train_y), (test_x, test_y)