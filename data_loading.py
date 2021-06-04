import numpy as np
import os
import glob
from scipy.io import loadmat
import matplotlib.pyplot as plt

def dataDir():
    return 'C://Users/ts1454/Projects/Bridge/Trial_1/September'


def loadTrial1Data(direction, count, data_set='Test Samples'):

    data_folder = os.path.join(dataDir(), data_set,
                        '{}_{:d}'.format(direction, count))
    mat_files = glob.glob(data_folder + '/*.mat')
    accel_data = []

    for file in mat_files:
        mat_data = loadmat(file)
        accel_data.append(mat_data['responses'][:,3] - mat_data['responses'][:,3].mean())

    accel_data = np.array(accel_data)
    print('{}, {} vehicle(s) {} data: '.format(direction, count, data_set), accel_data.shape)

    return accel_data


def normalise(x):
    y = 2 * (x - x.min())/(x.max() - x.min()) - 1
    #y = x / abs(x).max()
    return y


def moveRMS(x, window=30):

    x_length = x.shape[0]
    #y = np.zeros(x.shape)
    y = []

    for i in range(x_length):

        wi = min([i, window, 8000-i])
        #(wi)
        window_start = i - wi
        window_end = i + wi 

        window_data = x[window_start : window_end]
        RMS = np.sqrt(np.square(window_data).mean())


        y.append(RMS)
    print(i)
    return np.array(y)


def dataProcessing(accelData):
    
    norm = normalise(accelData)
    smooth = moveRMS(norm)

    offset = smooth[smooth < 0.1].mean()
    smooth = smooth - offset
    smooth = np.nan_to_num(smooth)

    return smooth
