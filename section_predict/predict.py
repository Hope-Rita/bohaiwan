import numpy as np
import platform

from utils.config import Config
config_path = '../section_predict/config.json'
conf = Config(config_path)

import utils.pred_utils as pu
from baseline import lr
from baseline import mlp
from baseline import recurrent
from baseline import svr
from baseline import xgb
from baseline import rf
from baseline import knn
from section_predict import gen_dataset
from utils import data_process
from utils import draw_pic
from utils import metric


def predict_one_section(data, func):
    """
    预测一个 section 的结果
    @param data: 预测用的数据
    @param func: 预测用的方法
    """
    pass


if __name__ == '__main__':
    pred_target = conf.get_config('predict-target')
    pred_target_filename = conf.get_data_loc(pred_target)
    pred_data = gen_dataset.load_one_section(filename=pred_target_filename, section_name='S01')
