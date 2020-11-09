import numpy as np
import platform


from utils.config import Config
config_path = '../section_predict/config.json'
conf = Config(config_path)
if conf.get_config('run-on-local'):
    pred_res_dir = conf.get_config('predict-result', 'local')
else:
    pred_res_dir = conf.get_config('predict-result', 'server')
pred_section = conf.get_config('predict-section')


import utils.pred_utils as pu
from baseline import lr
from baseline import mlp
from baseline import recurrent
from baseline import svr
from baseline import xgb
from baseline import rf
from baseline import knn
from section_predict import gen_dataset
from utils import normalization
from utils import data_process
from utils import draw_pic
from utils import metric


def predict_one_section(filename, data, func):
    """
    预测一个 section 的结果
    @param filename: 存放数据的文件
    @param data: 预测用的数据
    @param func: 预测用的方法
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
        # print(sensors[i], sensor_metric)
        section_metrics.append(sensor_metric)

    # 写入 CSV 文件。系统不同，处理方式不一样
    csv_name = func.__name__.split('_')[0] + f'_{gen_dataset.future_days}day' + '_' + pred_section + '_'
    csv_name += filename.split('\\')[-1] if platform.system() is 'Windows' else filename.split('/')[-1]

    data_process.dump_csv(pred_res_dir, csv_name, section_metrics, average_func=data_process.avg)


if __name__ == '__main__':
    pred_target = conf.get_config('predict-target')
    pred_target_filename = conf.get_data_loc(pred_target)
    pred_data = gen_dataset.load_one_section(filename=pred_target_filename, section_name=pred_section)
    predict_one_section(pred_target_filename, pred_data, recurrent.rnn_section_predict)
