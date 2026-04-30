import math


def rmse(predictions):
    mse = 0.0
    count = 0
    for pred in predictions:
        actual = pred[2]
        est = pred[3]
        mse += (actual - est) ** 2
        count += 1
    return math.sqrt(mse / count) if count else float('nan')
