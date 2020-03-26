#! /usr/bin/env python3

import numpy as np

def calcCost(input_WL_PIX, predicted_WL_PIX, delta_WL_PIX, numWLs, costType):
    if costType == "RMSE":
        err_WL_PIX = delta_WL_PIX
        meanSq_PIX = np.sum(err_WL_PIX * err_WL_PIX, axis=0) / numWLs
        d_r = np.sqrt(meanSq_PIX) / np.mean(input_WL_PIX, axis=0)	# normalise by input (?)
    elif costType == "RMSRE":
        err_WL_PIX = 1. - predicted_WL_PIX / input_WL_PIX
        meanSq_PIX = np.sum(err_WL_PIX * err_WL_PIX, axis=0) / numWLs
        d_r = np.sqrt(meanSq_PIX)
    elif costType == "RMSE_LOG":
        err_WL_PIX = np.log10(input_WL_PIX) - np.log10(predicted_WL_PIX)
        meanSq_PIX = np.sum(err_WL_PIX * err_WL_PIX, axis=0) / numWLs
        d_r = np.sqrt(meanSq_PIX)
    else:
        raise ValueError("invalid costType; must be one of 'RMSE', 'RMSRE', 'RMSE_LOG'")

    return d_r

# The '__main__' entry point is only used for testing. 
if __name__ == '__main__':
    import sys

    ME = sys.argv[0] 
    if len(sys.argv) != 1:
        print("usage:  " + sys.argv[0])
        sys.exit(1)

    # Define inputs:
    input_WL_PIX = np.arange(.1, .9, .1, dtype=float)
    input_WL_PIX.shape = (len(input_WL_PIX), 1)               # required by calcCost
    predicted_WL_PIX = input_WL_PIX * input_WL_PIX
    # predicted_WL_PIX.shape = (len(predicted_WL_PIX), 1)               # required by calcCost
    delta_WL_PIX = input_WL_PIX - predicted_WL_PIX
    numWLs = input_WL_PIX.shape[0]
    print("Inputs:")
    print("input_WL_PIX =", input_WL_PIX)
    print("predicted_WL_PIX =", predicted_WL_PIX)

    for costType in ['RMSE', 'RMSRE', 'RMSE_LOG']:
        cost_PIX = calcCost(input_WL_PIX, predicted_WL_PIX, delta_WL_PIX, numWLs, costType)
        print("\nResults for costType = " + costType + ":")
        print("cost_PIX =", cost_PIX)

    print(sys.argv[0] + " done")             # DEBUG
