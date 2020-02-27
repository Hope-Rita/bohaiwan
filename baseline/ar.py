from statsmodels.tsa.ar_model import AR


def ar_predict(x, future):
    """
    AR 模型
    :param x: 一维数组，x[t - pred_len + 1, ..., t - 1, t]
    :param future: 预测的天数
    :return: t + future 那天的预测值
    """
    model = AR(x)
    mf = model.fit(maxlag=len(x) - 1, trend='nc')
    pred = mf.predict(start=len(x), end=(len(x) + future - 1))
    return pred[-1]
