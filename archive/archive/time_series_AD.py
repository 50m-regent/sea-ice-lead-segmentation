
from utils.load_data import load_train_data, load_test_data

if __name__ == '__main__':
    train_x        = load_train_data()
    test_x, test_y = load_test_data()
    
    ad = AnomalyDetector()
    for 
    