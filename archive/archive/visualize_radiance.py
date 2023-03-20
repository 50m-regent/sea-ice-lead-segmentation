import cv2

from utils.readfiles import readfiles

DATA_PATH = 'data/S3B_OL_1_EFR____20190301T232521_20190301T232821_20200111T235148_0179_022_301_1800_MR1_R_NT_002.SEN3'
# DATA_PATH = 'data/S3A_OL_1_EFR____20180307T054004_20180307T054119_20180308T091959_0075_028_319_1620_LN1_O_NT_002.SEN3'

if __name__ == '__main__':
    dataset = readfiles(DATA_PATH)
    
    for name, data in dataset:
        cv2.imwrite('radiance/' + name + '.png', data)
        print(f"Saved: {name} {data.shape[0]}x{data.shape[1]}")