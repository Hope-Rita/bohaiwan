import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdate


from utils.config import Config
config_path = '../union_predict/config.json'
conf = Config(config_path)

from baseline import recurrent
from union_predict import gen_dataset
import utils.load_utils as ld
from utils import data_process


pred_target = conf.get_config('predict-target')
pred_target_filename = conf.get_data_loc(pred_target)
pred_col = conf.get_config('predict-col')
future_filename = conf.get_data_loc('taseometer-pro')
pred_func = recurrent.rnn_union_predict


def future_predict():
    train_x, train_y, train_date = ld.load_one_col(pred_target_filename,
                                                   pred_col,
                                                   gen_dataset.gen_data,
                                                   add_date=True,
                                                   split=False,
                                                   normalize=False
                                                   )
    test_x, pred_dates = gen_dataset.future_dataset(future_filename, pred_col)

    test_dates = [d - pd.Timedelta(days=conf.get_config('data-parameters', 'future-days')) for d in pred_dates]
    observe_val = np.array([ts[-4] for ts in test_x])

    tmp = data_process.col_normalization(np.concatenate((train_x, test_x)))
    train_x = tmp[:len(train_x)]
    test_x = tmp[len(train_x):]

    train_x = np.array(train_x)
    train_y = np.array(train_y)
    test_x = np.array(test_x)

    pred = pred_func(train_x, train_y, test_x)
    print(pred)
    print(pred_dates)
    print(observe_val)
    data_process.dump_pred_result('future_pred/', f'{pred_col}.csv', None,
                                  np.concatenate((observe_val, pred)), test_dates + pred_dates)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d'))
    plt.xticks(pd.date_range(test_dates[0], pred_dates[-1]), rotation=90)
    plt.plot(test_dates + pred_dates, np.concatenate((observe_val, pred)), marker='o')
    plt.plot(test_dates + pred_dates,  np.concatenate((np.array([np.nan] * len(test_dates)), pred)), marker='o')
    plt.title(pred_col)
    plt.show()


def variation_paras(para, variation_range):
    train_x, train_y, train_date = ld.load_one_col(pred_target_filename,
                                                   pred_col,
                                                   gen_dataset.gen_data,
                                                   add_date=True,
                                                   split=False,
                                                   normalize=False
                                                   )
    test_x, pred_dates = gen_dataset.future_dataset(future_filename, pred_col)
    variation_range = list(variation_range)

    change_idx = -1 if para == 'waterline' else (-2 if para == 'low_tp' else -3)

    var_input = []
    for vr in variation_range:
        tmp = list(test_x[-1])
        tmp[change_idx] = vr
        var_input.append(tmp)

    var_input = np.array(var_input)

    tmp = data_process.col_normalization(np.concatenate((train_x, var_input)))
    train_x = tmp[:len(train_x)]
    var_input = tmp[len(train_x):]

    pred = pred_func(train_x, train_y, var_input)
    print(variation_range)
    print(pred)
    plt.plot(variation_range, pred, marker='o')
    plt.title(pred_col + ' ' + para)
    plt.show()


if __name__ == '__main__':
    # future_predict()
    v1 = [2 + 0.2 * i for i in range(51)]
    variation_paras('waterline', v1)
    v2 = [24 + 0.5 * i for i in range(25)]
    variation_paras('high_tp', v2)
    v3 = [18 + 0.5 * i for i in range(25)]
    variation_paras('low_tp', v3)
