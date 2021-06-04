import numpy as np
import matplotlib.pyplot as plt
from data_loading import dataProcessing 
from scipy.stats import lognorm
from scipy.optimize import minimize, shgo, dual_annealing


def logNormFit(params, t):
    
    mu = params[0]
    sigma = params[1]
    scale = params[2]
    shift = params[3]

    fit = scale * lognorm.pdf(t-shift,1,loc=0)
    return fit


def objectiveFunc(params, accelData, t):

    fit = logNormFit(params, t)

    above_floor = accelData > 0

    return (np.square(fit[above_floor] - accelData[above_floor])).mean()


def fitCurve(accelData,t):

    x0 = [1,1,1,1]
    bounds = [(0,10),(0,10),(0,10),(0,120)]

    result = dual_annealing(objectiveFunc, args=(accelData,t), bounds = bounds)
    print(result.x)
    fit = logNormFit(result.x, t)


    plt.plot(t, accelData)
    plt.plot(t, fit)
    plt.show()




def estimateVehicles(accelData):
    # one by one
    time = np.linspace(0,8000/64,8000)
    # pre processing
    accelData = dataProcessing(accelData)
    # method
    fitCurve(accelData, time)