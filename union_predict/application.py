import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdate
from importlib import reload


from utils.config import Config
config_path = '../union_predict/config.json'
conf = Config(config_path)

from baseline import recurrent
from union_predict import gen_dataset
import utils.load_utils as ld
from utils import data_process
from utils import normalization


pred_target = conf.get_config('predict-target')
pred_target_filename = conf.get_data_loc(pred_target)
pred_col = conf.get_config('predict-col')
pred_func = recurrent.rnn_union_predict
temperature_bias = 3.715


def future_predict():
    # 方案1：固定 q，历史数据变化
    train_x, train_y, train_date = ld.load_one_col(pred_target_filename,
                                                   pred_col,
                                                   gen_dataset.gen_data,
                                                   add_date=True,
                                                   split=False,
                                                   normalize=False
                                                   )
    test_x, pred_dates = gen_dataset.future_dataset(pred_target_filename, pred_col)

    test_dates = [d - pd.Timedelta(days=conf.get_config('data-parameters', 'future-days')) for d in pred_dates]
    observe_val = np.array([ts[-4] for ts in test_x])

    train_x = np.array(train_x)
    train_x, normalizer = data_process.col_normalization_with_normalizer(train_x)
    test_x = data_process.col_transform(test_x, normalizer)
    # train_x = normalization.sigmoid(train_x)
    # test_x = normalization.sigmoid(test_x)

    # 进行预测
    pred = pred_func(train_x, train_y, test_x)
    print(pred)
    print(pred_dates)
    print(observe_val)
    data_process.dump_pred_result('future_pred/', f'{pred_col}.csv', None,
                                  np.concatenate((observe_val, pred)), test_dates + pred_dates)
    plot_future_trend(test_dates, pred_dates, observe_val, pred)


def future_predict2():
    # 历史数据固定，q变化
    dates = []
    preds = []
    observe_val = None

    for q in range(1, 15):
        print('q =', q)

        conf.modify_config('data-parameters', 'future-days', new_val=q)

        # 重新 import 更新 pred-len 参数
        reload(gen_dataset)
        reload(recurrent)

        train_x, train_y, train_date = ld.load_one_col(pred_target_filename,
                                                       pred_col,
                                                       gen_dataset.gen_data,
                                                       add_date=True,
                                                       split=False,
                                                       normalize=False
                                                       )
        test_x, pred_dates = gen_dataset.future_dataset(pred_target_filename, pred_col)
        # 取最后一天
        test_x = np.array([test_x[-1]])
        dates.append(pred_dates[-1])
        if observe_val is None:
            observe_val = test_x[-1][:-conf.get_config('data-parameters', 'env-factor-num')]

        tmp = data_process.col_normalization(np.concatenate((train_x, test_x)))
        train_x = tmp[:len(train_x)]
        test_x = tmp[len(train_x):]

        train_x = np.array(train_x)
        train_y = np.array(train_y)
        test_x = np.array(test_x)

        pred = pred_func(train_x, train_y, test_x)
        preds.append(pred[-1])
        print()

    test_dates = [d - pd.Timedelta(days=conf.get_config('data-parameters', 'future-days')) for d in dates]
    data_process.dump_pred_result('future_pred/', f'{pred_col}.csv', None,
                                  np.concatenate((observe_val, preds)), test_dates + dates)
    plot_future_trend(test_dates=test_dates, pred_dates=dates, observe_val=observe_val, pred=preds)


def variation_paras(para, variation_range):
    original_train_x, train_y, train_date = ld.load_one_col(pred_target_filename,
                                                   pred_col,
                                                   gen_dataset.gen_data,
                                                   add_date=True,
                                                   split=False,
                                                   normalize=False
                                                   )
    test_x, pred_dates = gen_dataset.future_dataset(pred_target_filename, pred_col)
    variation_range = list(variation_range)

    change_idx = -1 if para == 'waterline' else (-2 if para == 'low_tp' else -3)

    res = []
    for vr in variation_range:
        input_vec = list(test_x[-1])
        input_vec[change_idx] = vr
        input_vec = np.array([input_vec])

        tmp = data_process.col_normalization(np.concatenate((original_train_x, input_vec)))
        train_x = tmp[:len(original_train_x)]
        var_input = tmp[len(original_train_x):]

        pred = pred_func(train_x, train_y, var_input)
        res.append(pred[-1])

    print(variation_range)
    print(res)
    plt.plot(variation_range, res, marker='o')
    plt.title(pred_col + ' ' + para)
    plt.show()


def variation_paras2(temperature_range, waterline_range):

    # train_x 要存一份用来保证每次的输入
    original_train_x, train_y, train_date = ld.load_one_col(pred_target_filename,
                                                   pred_col,
                                                   gen_dataset.gen_data,
                                                   add_date=True,
                                                   split=False,
                                                   normalize=False
                                                   )
    test_x, pred_dates = gen_dataset.future_dataset(pred_target_filename, pred_col)

    res = {}

    for t in temperature_range:
        print(f'Average temperature:{t}℃')

        var_input = []
        for w in waterline_range:
            tmp = list(test_x[-1])
            tmp[-1] = w
            tmp[-2] = t - temperature_bias
            tmp[-3] = t + temperature_bias
            var_input.append(tmp)

        var_input = np.array(var_input)

        tmp = data_process.col_normalization(np.concatenate((original_train_x, var_input)))
        train_x = tmp[:len(original_train_x)]
        var_input = tmp[len(train_x):]

        pred = pred_func(train_x, train_y, var_input)
        res[t] = pred

    res_df = pd.DataFrame(res, index=waterline_range)
    res_df.to_csv(f'variation_para/multiple/{pred_col}.csv', index_label='waterline')
    hot_map(res_df.to_numpy(),
            xticks=temperature_range,
            yticks=waterline_range,
            xlabel='Daily average temperature',
            ylabel='Waterline',
            pic_title=pred_col
            )


def variation_paras3(para, variation_range):
    train_x, train_y, train_date = ld.load_one_col(pred_target_filename,
                                                   pred_col,
                                                   gen_dataset.gen_data,
                                                   add_date=True,
                                                   split=False,
                                                   normalize=False
                                                   )
    test_x, pred_dates = gen_dataset.future_dataset(pred_target_filename, pred_col)
    variation_range = list(variation_range)

    change_idx = -1 if para == 'waterline' else (-2 if para == 'low_tp' else -3)

    var_input = []
    for vr in variation_range:
        tmp = list(test_x[-1])
        tmp[change_idx] = vr
        var_input.append(tmp)

    var_input = np.array(var_input)
    train_x = np.array(train_x)
    train_x, normalizer = data_process.col_normalization_with_normalizer(train_x)
    var_input = data_process.col_transform(var_input, normalizer)
    # var_input = normalization.sigmoid(var_input)

    pred = pred_func(train_x, train_y, var_input)
    print(variation_range)
    print(pred)
    plt.plot(variation_range, pred, marker='o')
    plt.title(pred_col + ' ' + para)
    plt.show()


def plot_future_trend(test_dates, pred_dates, observe_val, pred):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d'))
    plt.xticks(pd.date_range(test_dates[0], pred_dates[-1]), rotation=90)
    plt.plot(test_dates + pred_dates, np.concatenate((observe_val, pred)), marker='o')
    plt.plot(test_dates + pred_dates, np.concatenate((np.array([np.nan] * len(test_dates)), pred)), marker='o')
    plt.title(pred_col)
    plt.show()


def hot_map(matrix, xticks, yticks, xlabel, ylabel, pic_title):
    plt.imshow(matrix, cmap='YlGnBu')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.colorbar()
    plt.xticks(range(0, len(xticks), 2), [xticks[i] for i in range(len(xticks)) if i % 2 == 0], rotation=90)
    plt.yticks(range(0, len(yticks), 2), [yticks[i] for i in range(len(yticks)) if i % 2 == 0])
    plt.title(pic_title)
    plt.show()


if __name__ == '__main__':
    # future_predict()
    # future_predict2()
    v1 = [2 + 0.25 * i for i in range(41)]
    # v1 = [2 + 0.25 * i for i in range(13)]
    # variation_paras('waterline', v1)
    v2 = [24 + 0.5 * i for i in range(25)]
    # variation_paras('high_tp', v2)
    v3 = [18 + 0.5 * i for i in range(25)]
    # variation_paras('low_tp', v3)

    v4 = [18.715 + 0.5 * i for i in range(34)]
    # v4 = [29.715 + 0.5 * i for i in range(34)]
    # v4 = [20.215 + 0.5 * i for i in range(33)]
    # v4 = [18 + 0.5 * i for i in range(39)]
    # variation_paras2(temperature_range=v4, waterline_range=v1)

    conf.modify_config('model-parameters', 'recurrent', 'load-model', new_val=False)
    conf.modify_config('model-parameters', 'recurrent', 'save-model', new_val=True)
    reload(recurrent)
    future_predict()
    conf.modify_config('model-parameters', 'recurrent', 'load-model', new_val=True)
    conf.modify_config('model-parameters', 'recurrent', 'save-model', new_val=False)
    reload(recurrent)
    # variation_paras2(temperature_range=v4, waterline_range=v1)
    variation_paras3('waterline', v1)
    # variation_paras('high_tp', v2)
    variation_paras3('high_tp', v2)
    variation_paras3('low_tp', v3)
    variation_paras2(temperature_range=v4, waterline_range=v1)
