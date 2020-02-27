import numpy as np
from baseline import ar
from baseline import ha
from baseline import lr
from baseline import lstm
from baseline import mlp
from baseline import svr
from baseline import xgb
from single_predict import gen_dataset
from utils import draw_pic
from utils import metric
from utils.config import get_config


col = 'FSS_S01_B1_003'
# col = 'FCS_S01_B2_006'


def ar_model(filename):
    x_train, y_train, x_test, y_test = gen_dataset.load_all(filename)
    # x_test, y_test = gen_dataset.gen_data(filename, col)

    pred = []
    for i in range(len(x_test)):
        pred.append(ar.ar_predict(x_test[i], gen_dataset.future_days))
    pred = np.array(pred)

    print(metric.all_metric(y_test, pred))


def history_average(filename):
    x_train, y_train, x_test, y_test = gen_dataset.load_all(filename)

    pred = []
    for i in range(len(x_test)):
        pred.append(ha.ha_predict(x_test[i]))
    pred = np.array(pred)

    print(metric.all_metric(y_test, pred))


def supervised_model(filename, func, one_col=False):

    if one_col:
        x, y = gen_dataset.gen_data(filename, col)
        train_size = int(0.7 * len(x))
        x_train, y_train = x[:train_size], y[:train_size]
        x_test, y_test = x[train_size:], y[train_size:]
    else:
        x_train, y_train, x_test, y_test = gen_dataset.load_all(filename)

    pred = func(x_train, y_train, x_test)
    print(func.__name__, filename, gen_dataset.future_days)
    print(metric.all_metric(y_test, pred))


def all_supervised_models(filename):
    x_train, y_train, x_test, y_test = gen_dataset.load_all(filename)
    for func in [lr.lr_predict, svr.svr_predict, xgb.xgb_predict, mlp.mlp_predict]:
        pred = func(x_train, y_train, x_test)
        print(func.__name__, filename, gen_dataset.future_days)
        print(metric.all_metric(y_test, pred))


def predict_plot(filename):
    x, y = gen_dataset.gen_data(filename, col)
    train_size = int(0.7 * len(x))
    x_train, y_train, x_test, y_test = gen_dataset.load_all(filename)
    x_test, y_test = x[train_size:], y[train_size:]

    d = dict()
    d['True value'] = y
    d['LR'] = lr.lr_predict(x_train, y_train, x_test)
    d['SVR'] = svr.svr_predict(x_train, y_train, x_test)
    d['XGBoost'] = xgb.xgb_predict(x_train, y_train, x_test)
    d['MLP'] = mlp.mlp_predict(x_train, y_train, x_test)

    ha_pred = []
    for i in range(len(x_test)):
        ha_pred.append(ha.ha_predict(x_test[i]))
    d['HA'] = np.array(ha_pred)

    ar_pred = []
    for i in range(len(x_test)):
        ar_pred.append(ar.ar_predict(x_test[i], gen_dataset.future_days))
    d['AR'] = np.array(ar_pred)
    draw_pic.all_predict(d)


if __name__ == '__main__':
    pred_target = get_config('config.json', 'predict-target')
    pred_target_filename = get_config('../data/data.json', pred_target, 'local')
    supervised_model(pred_target_filename, lstm.lstm_predict)
