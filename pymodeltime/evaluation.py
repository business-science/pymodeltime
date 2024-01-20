

def mae(y_true, y_pred):
    return (y_true - y_pred).abs().mean()

def rmse(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean() ** 0.5

def mape(y_true, y_pred):
    return ((y_true - y_pred).abs() / y_true.abs()).mean()

def smape(y_true, y_pred):
    denominator = (y_true.abs() + y_pred.abs())
    diff = (y_true - y_pred).abs() / denominator
    diff[denominator == 0] = 0  # handle the case where the denominator is zero
    return 2 * diff.mean()
