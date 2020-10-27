# fitness evaluation
import numpy as np


def evaluate(test, predict):
    predicted_temp, predicted_mlr = predict['Temp'], predict['mlr']
    test_temp, test_mlr = test['Temp'], predict['mlr']
    cost = 1/(np.sum(np.square(predicted_temp - test_temp))
              + np.square(predicted_mlr.max() - test_mlr.max()))
    return cost
