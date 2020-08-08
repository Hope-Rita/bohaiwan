import matplotlib.pyplot as plt
import pandas as pd
import os
from importlib import reload


from utils.config import Config
config_path = '../union_predict/pred_len_survey.json'
conf = Config(config_path)


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


def produce_result(func, repeat_id=None):
    # 跑多次试验，收集结果
    res = []
    for pred_len in range(5, 21):
        # mlp.hidden_size = (pred_len + 1) // 2
        # mlp.hidden_size = tuple([pred_len * 2, pred_len, pred_len // 2])
        res.append(k_day_predict(func, pred_len))

    result_frame = merge_result(res)

    # 写入 CSV
    if repeat_id:
        result_frame.to_csv(f'pred_len_survey/metrics/{func.__name__}_repeat{repeat_id}.csv',
                            index=True,
                            index_label='Column'
                            )
    else:
        result_frame.to_csv(f'pred_len_survey/metrics/{func.__name__}.csv', index=True, index_label='Column')


def k_day_predict(pred_func, k):
    """
    执行 pred-len 为 k 的预测
    :param pred_func: 预测使用的模型
    :param k: 使用 t 时刻的前 k 天当作 pred-len
    :return: 模型产生的指标，类型为 DataFrame
    """
    # 修改 config
    conf.modify_config('data-parameters', 'pred-len', new_val=k)

    # 重新 import 更新 pred-len 参数
    reload(gen_dataset)
    reload(recurrent)

    print('当前的 pred-len 为', gen_dataset.pred_len)
    # 载入数据
    pred_target = conf.get_config('predict-target')
    pred_target_filename = conf.get_data_loc(pred_target)
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


def avg_metric_plot(frame, keyword, model_ame):
    key_cols = [col for col in frame.columns if keyword in col]

    x = []
    y = []
    for col in key_cols:
        x.append(int(col[:col.index('d')]))
        y.append(data_process.avg(frame.loc[:, col]))

    plt.cla()
    plt.plot(x, y)
    plt.title(keyword)
    plt.savefig(f'pred_len_survey/pics/{model_ame}_{keyword}.png')


def avg_box_plot(func, repeat_num):
    """
    根据多次重复实验的结果绘制箱线图
    :param func: 使用的预测模型
    :param repeat_num: 重复的次数
    """

    def draw_box(keyword):
        key_cols = [col for col in avg_frame.columns if keyword in col]
        key_frame = avg_frame[key_cols]
        plt.boxplot(key_frame.to_numpy(), labels=[col[:col.index('d')] for col in key_cols])
        plt.title(f'{func.__name__}_{keyword}')
        plt.show()

    # 对每一次重复实验，取各列的平均值
    avg_list = []
    for i in range(1, repeat_num + 1):
        filename = f'pred_len_survey/metrics/{func.__name__}_repeat{i}.csv'
        df = pd.read_csv(filename, index_col='Column')
        avg_list.append(df.mean(axis=0))
    # 压缩成一个 DataFrame，每行代表一次重复实验
    avg_frame = pd.DataFrame(avg_list)

    draw_box('RMSE')
    draw_box('PCC')
    draw_box('MAPE')


def avg_col_trend(func, repeat_num):
    """
    进行多次重复实验，并绘制每个传感器预测效果随着 pred_len 变化的平均趋势图
    :param func: 使用的预测模型
    :param repeat_num: 重复实验的次数
    """
    def draw_plot(series, name, keyword):
        key_series = series[[col for col in series.index if keyword in col]]
        plt.plot([idx[:idx.index('d')] for idx in key_series.index], key_series.to_numpy(), marker='o')
        plt.title(f'{name}_{keyword}')
        plt.savefig(os.path.join(f'pred_len_survey/col_trend/{func.__name__}', f'{name}_{keyword}'))
        plt.clf()

    # 遍历所有的实验记录，把同个传感器的记录放到一起
    sensor_series = {}
    for i in range(1, repeat_num + 1):
        filename = f'pred_len_survey/metrics/{func.__name__}_repeat{i}.csv'
        df = pd.read_csv(filename, index_col='Column')

        for sensor_name, row in df.iterrows():
            if sensor_name not in sensor_series:
                sensor_series[sensor_name] = []
            sensor_series[sensor_name].append(row)

    # 对每个传感器的记录求均值，并进行绘图
    for sensor_name in sensor_series:
        sensor_frame = pd.DataFrame(sensor_series[sensor_name])
        sensor_avg = sensor_frame.mean(axis=0)

        draw_plot(sensor_avg, sensor_name, 'RMSE')
        draw_plot(sensor_avg, sensor_name, 'PCC')


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


def repeat_run(start, end, func):
    """
    多次运行取平均值，这里使用范围表示重复运行的范围，[start, end)
    """
    for repeat_id in range(start, end):
        print(f'repeat-id {repeat_id}')
        produce_result(func, repeat_id)


if __name__ == '__main__':

    pred_model = recurrent.rnn_union_predict
    repeat_run(20, 21, pred_model)
    # avg_box_plot(pred_model, 20)
    # avg_col_trend(pred_model, 20)
