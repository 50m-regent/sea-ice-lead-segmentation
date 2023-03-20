from utils.load_data import load_test_data

import sys

if __name__ == '__main__':
    n = int(sys.argv[1])
    
    test_x, test_y = load_test_data()
    
    s = 0
    sc = 0
    t = 0
    tc = 0
    for x, y in zip(test_x, test_y):
        if 0 == y:
            print(f'{x[n]:>.5f} {int(y)}')
            s += x[n]
            sc += 1
        else:
            print(f'{x[n]:>.5f} {int(y)}')
            t += x[n]
            tc += 1
    
    print(s / sc, t / tc, s / sc - t / tc)
    