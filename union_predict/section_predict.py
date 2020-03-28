"""
使用整个 section 的时间序列，再加上环境因素作为数据集进行测试
采取两种方案来预测
1）分列进行预测
2）整个 section 一起预测
"""


import utils.pred_utils as pu
from baseline import lr
from baseline import mlp
from baseline import recurrent
from baseline import svr
from baseline import xgb
from union_predict import gen_dataset
from utils import data_process
from utils.config import get_config


# 存放预测结果文件的路径
res_dir1, res_dir2 = get_config('section_config.json', 'predict-result', inner_keys=['result1', 'result2'])


def scheme1(filename, section):
    """
    方案1：对某个 section 上的所有列数据分别进行预测
    :param filename: 存放数据的 CSV 文件
    :param section: section 的名字
    """

    # 加载给定 section 上的数据集
    data = gen_dataset.load_section(filename, section)

    for func in [lr.lr_pca_predict, svr.svr_predict, xgb.xgb_predict, mlp.mlp_predict]:
        predict_one_cols(func, data, filename)


def predict_one_cols(func, data, filename):
    """
    用给定的模型对每一列的数据分别进行预测
    :param func: 使用的模型
    :param data: 预测使用的数据，格式为字典
    :param filename: 存放数据的文件
    """
    print('模型：', func.__name__, 'future_days:', gen_dataset.future_days)

    # 进行训练，得到每一列数据的预测指标
    cols_metrics = pu.predict_one_cols(func, data)

    # 写入 CSV 文件
    csv_name = func.__name__.split('_')[0] + f'_{gen_dataset.future_days}day' + '_' + filename.split('/')[-1]
    data_process.dump_csv(res_dir1, csv_name, cols_metrics, avg=data_process.avg)


if __name__ == '__main__':
    pred_target = get_config('section_config.json', 'predict-target')
    pred_target_filename = get_config('../data/data.json', pred_target, 'server')
    # scheme1(pred_target_filename, 'S01')

    pred_data = gen_dataset.load_section(pred_target_filename, 'S01')
    predict_one_cols(recurrent.lstm_section_predict, pred_data, pred_target_filename)
