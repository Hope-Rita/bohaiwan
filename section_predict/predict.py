from utils.config import Config
config_path = '../section_predict/config.json'
conf = Config(config_path)
if conf.get_config('run-on-local'):
    pred_res_dir = conf.get_config('predict-result', 'local')
else:
    pred_res_dir = conf.get_config('predict-result', 'server')
pred_section = conf.get_config('predict-section')


import platform
import os
import pandas as pd
from importlib import reload
from baseline import recurrent
from section_predict import gen_dataset
from utils import normalization
from utils import data_process
from utils import metric


def adjust_para(filename, data, func):
    """
    使用不同的隐藏层规模和学习率进行实验，得到最佳的组合
    @param filename: 存放数据的文件名
    @param data: 数据
    @param func: 使用的模型
    """
    hidden_size_range = range(5, 85, 5)
    lr_range = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]

    res = []
    for hidden_size in hidden_size_range:
        recurrent_model = func.__name__.split('_')[0]
        conf.modify_config('model-parameters', 'recurrent', f'{recurrent_model}-hidden-size', new_val=hidden_size)
        metrics = {'hidden_size': hidden_size}

        for lr in lr_range:
            print(f'model: {func.__name__}, hidden_size: {hidden_size}, lr: {lr}')
            conf.modify_config('model-parameters', 'recurrent', 'learning-rate', new_val=lr)
            reload(recurrent)
            pred_res = predict_one_section(filename, data, func, dump_csv=False)
            df = pd.DataFrame(pred_res)
            metrics[lr] = data_process.avg(df.loc[:, 'RMSE'])
            print()

        res.append(metrics)

    csv_name = func.__name__.split('_')[0] + f'_{gen_dataset.future_days}day' + '_' + pred_section + '_'
    csv_name += filename.split('\\')[-1] if platform.system() is 'Windows' else filename.split('/')[-1]
    res_frame = pd.DataFrame(res)
    res_frame.to_csv(path_or_buf=os.path.join('adjust_para', csv_name), index=False)


def predict_one_section(filename, data, func, dump_csv=True):
    """
    预测一个 section 的结果
    @param filename: 存放数据的文件
    @param data: 预测用的数据
    @param func: 预测用的方法
    @param dump_csv: 是否将预测结果的指标写入到 CSV 中
    @return 该 section 内每个传感器的预测指标，格式为 List(dict)
    """
    x_train, y_train, x_test, y_test, sensors = data

    # 对 y 进行归一化
    normal_y = normalization.MinMaxNormal([y_train, y_test])
    y_train = normal_y.transform(y_train)

    pred = func(x_train, y_train, x_test)
    pred = normal_y.inverse_transform(pred)
    section_metrics = []

    for i in range(y_test.shape[1]):

        sensor_metric = {'Column': sensors[i]}

        sensor_metric.update(metric.all_metric(y_test[:, i], pred[:, i]))
        section_metrics.append(sensor_metric)

    if dump_csv:  # 写入 CSV 文件。系统不同，处理方式不一样
        csv_name = func.__name__.split('_')[0] + f'_{gen_dataset.future_days}day' + '_' + pred_section + '_'
        csv_name += filename.split('\\')[-1] if platform.system() is 'Windows' else filename.split('/')[-1]
        data_process.dump_csv(pred_res_dir, csv_name, section_metrics, average_func=data_process.avg)

    return section_metrics


if __name__ == '__main__':
    pred_target = conf.get_config('predict-target')
    pred_target_filename = conf.get_data_loc(pred_target)
    pred_data = gen_dataset.load_one_section(filename=pred_target_filename, section_name=pred_section)
    # predict_one_section(pred_target_filename, pred_data, recurrent.rnn_section_predict)
    adjust_para(pred_target_filename, pred_data, recurrent.lstm_section_predict)
