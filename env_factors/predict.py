import platform


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


if __name__ == '__main__':
    pred_target = conf.get_config('predict-target')
    pred_target_filename = conf.get_data_loc(pred_target)
    run_all_models(pred_target_filename)
