import math

def rmse(y_true, y_pred):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(y_true, y_pred)) / len(y_true))

def mae(y_true, y_pred):
    return sum(abs(a - b) for a, b in zip(y_true, y_pred)) / len(y_true)

def mape(y_true, y_pred):
    eps = 1e-12
    return 100 * sum(abs((a - b) / (a if abs(a) > eps else eps)) for a, b in zip(y_true, y_pred)) / len(y_true)

def r2(y_true, y_pred):
    mean_y = sum(y_true) / len(y_true)
    ss_res = sum((a - b) ** 2 for a, b in zip(y_true, y_pred))
    ss_tot = sum((a - mean_y) ** 2 for a in y_true)
    return 1 - ss_res / ss_tot if ss_tot != 0 else 0
