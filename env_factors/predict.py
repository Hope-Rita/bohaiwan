import platform
import pandas as pd

from utils.config import Config
conf = Config('config.json')

import utils.pred_utils as pu
from baseline import lr
from baseline import mlp
from baseline import svr
from baseline import xgb
from env_factors import gen_dataset
from utils import data_process


# 存放预测结果的文件路径
result_dir = conf.get_config('predict-result', 'result')


def run_all_models(filename):
    """
    方案3：采用多个模型分别对每一列进行训练和测试
    :param filename: 存放数据的文件
    """
    data = gen_dataset.load_cols(filename)

    for func in [lr.lr_predict, svr.svr_predict, xgb.xgb_predict, mlp.mlp_predict]:
        predict_one_cols(func, data, filename)


def predict_one_cols(func, data, filename):
    """
    用给定的模型对每一列的数据分别进行预测
    :param func: 使用的模型
    :param data: 预测使用的数据，格式为字典
    :param filename: 存放数据的文件
    """
    print(f'模型: {func.__name__}')

    # 进行训练，得到每一列数据的预测指标
    cols_metrics = pu.predict_one_cols(func, data)

    # 写入 CSV 文件。系统不同，处理方式不一样
    csv_name = func.__name__.split('_')[0] + '_'
    csv_name += filename.split('\\')[-1] if platform.system() is 'Windows' else filename.split('/')[-1]

    data_process.dump_csv(result_dir, csv_name, cols_metrics, average_func=data_process.avg)


def cross_validation(filename, func, k=10):
    """
    对每一列传感器进行多折交叉验证，输出各列的值，并绘制图像
    :param filename: 存放数据的文件
    :param func: 使用的预测模型
    :param k: 折数
    """
    frame = pd.read_csv(pred_target_filename, parse_dates=True, index_col='date')
    cols = gen_dataset.get_valid_cols(frame.columns)

    col_metrics = []
    for col in cols:
        print('当前列：', col)
        metric_dict = one_col_cross_validation(filename, col, func, k)
        metric_dict['Column'] = col
        col_metrics.append(metric_dict)
        print()

    # 写入 CSV 文件。系统不同，处理方式不一样
    csv_name = func.__name__.split('_')[0] + '_'
    csv_name += filename.split('\\')[-1] if platform.system() is 'Windows' else filename.split('/')[-1]

    data_process.dump_csv(f'col_result/{func.__name__}', csv_name, col_metrics, average_func=data_process.avg)


def one_col_cross_validation(filename, col, func, k=10, is_draw_pic=True):
    """
    对某一列传感器的数据进行多折交叉验证
    :param filename: 存放数据的文件名
    :param col: 当前列号
    :param func: 使用的预测模型
    :param k: 折数
    :param is_draw_pic: 是否绘制图像
    :return: 这列的预测指标
    """
    x, y, date = gen_dataset.load_one_col_not_split(filename, col, add_date=True)
    return pu.one_col_cross_validation((x, y), date, func, k, is_draw_pic,
                                       csv_loc={
                                           'dir': f'col_result/{func.__name__}/vals',
                                           'filename': f'{col}.csv'
                                       },
                                       pic_info={
                                           'dir': f'col_result/{func.__name__}/pics',
                                           'filename': f'{col}.jpg',
                                           'title': col
                                       })


if __name__ == '__main__':
    pred_target = conf.get_config('predict-target')
    pred_target_filename = conf.get_data_loc(pred_target)

    # run_all_models(pred_target_filename)
    cross_validation(pred_target_filename, xgb.xgb_predict)