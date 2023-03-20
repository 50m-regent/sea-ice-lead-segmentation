def split_data(x, train_ratio = 0.8):
    assert 0 <= train_ratio <= 1
    
    i = int(len(x) * train_ratio)
    return x[:i], x[i:]
    