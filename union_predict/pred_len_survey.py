import matplotlib.pyplot as plt
import pandas as pd
import utils.pred_utils as pu
from utils.config import *
from baseline import lr
from baseline import recurrent
from baseline import mlp
from baseline import svr
from baseline import xgb
from union_predict import predict
from utils import data_process
from union_predict import gen_dataset
from importlib import reload


def produce_result(func):
    # 跑多次试验，收集结果
    res = []
    for pred_len in range(5, 21):
        # mlp.hidden_size = (pred_len + 1) // 2
        # mlp.hidden_size = tuple([pred_len * 2, pred_len, pred_len // 2])
        res.append(k_day_predict(func, pred_len))

    result_frame = merge_result(res)

    # 写入 CSV
    result_frame.to_csv('pred_len_survey.csv', index=True, index_label='Column')


def k_day_predict(pred_func, k):
    """
    执行 pred-len 为 k 的预测
    :param pred_func: 预测使用的模型
    :param k: 使用 t 时刻的前 k 天当作 pred-len
    :return: 模型产生的指标，类型为 DataFrame
    """
    # 修改 config
    global_config.modify_config('data-parameters', 'pred-len', new_val=k)

    # 重新 import 更新 pred-len 参数
    reload(gen_dataset)
    reload(recurrent)

    print('当前的 pred-len 为', gen_dataset.pred_len)
    # 载入数据
    pred_target = global_config.get_config('predict-target')
    pred_target_filename = global_config.get_data_loc(pred_target)
    data = gen_dataset.load_cols(pred_target_filename)

    # 测试模型, 返回结果
    result = pu.predict_one_cols(pred_func, data)
    return assemble_frame(result, k_day=k)


def metric_plot(frame, keyword):

    plt.figure(figsize=(15, 8))
    key_cols = [col for col in frame.columns if keyword in col]

    for col in key_cols:
        plt.plot(frame.loc[:, col].to_numpy(), label=col.split('_')[0])

    plt.legend()
    plt.title(keyword)
    plt.savefig(f'pred_len_survey_{keyword}.jpg')


def avg_metric_plot(frame, keyword):
    key_cols = [col for col in frame.columns if keyword in col]

    x = []
    y = []
    for col in key_cols:
        x.append(int(col[:col.index('d')]))
        y.append(data_process.avg(frame.loc[:, col]))

    plt.cla()
    plt.plot(x, y)
    plt.title(keyword)
    plt.savefig(f'pred_len_survey_{keyword}.png')


def merge_result(frame_list):
    """
    在 DataFrame 的层面进行合并
    :param frame_list: 存放 DataFrame 的列表
    :return: 合并后的列表，index 为列号
    """
    frame = frame_list[0]
    for i in range(1, len(frame_list)):
        frame = pd.merge(frame, frame_list[i], left_index=True, right_index=True)
    return frame


def assemble_frame(metric_list, k_day):
    """
    把得到的指标列表组装成 DataFrame
    :param metric_list: List[dict] 格式
    :param k_day: pred-len 天数，加在各个指标的名字前
    :return: index 为列号的 DataFrame
    """
    index = [d.pop('Column') for d in metric_list]  # 把 index 提取出来

    new_list = []
    for d in metric_list:

        new_dict = {}
        for k in d:  # 改变列名
            new_dict[f'{k_day}day_{k}'] = d[k]

        new_list.append(new_dict)

    del metric_list
    return pd.DataFrame(new_list, index=index)


if __name__ == '__main__':

    produce_result(recurrent.lstm_union_predict)

    # 绘图
    df = pd.read_csv('pred_len_survey.csv', index_col='Column')
    avg_metric_plot(df, 'MAPE')
    avg_metric_plot(df, 'PCC')
    avg_metric_plot(df, 'RMSE')
