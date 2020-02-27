"""
采取两种方案来预测
1）所有的数据一起训练，分列进行测试
2）分列进行训练和测试
"""


from baseline import lr
from baseline import lstm
from baseline import mlp
from baseline import svr
from baseline import xgb
from union_predict import gen_dataset
from utils import data_process
from utils import metric
from utils.config import get_config


# 存放预测结果文件的路径
res_dir1, res_dir2 = get_config('config.json', 'predict-result', inner_keys=['result1', 'result2'])


def predict_every_col(filename):
    # 方案 1
    x_train, y_train, test_data = gen_dataset.load_every_col(filename)

    for func in [lr, svr, xgb, mlp]:
        # 每个模型都训练一遍
        print('模型：', func.__name__)

        result_list = []
        model = func.model_fit(x_train, y_train)

        for key in test_data:
            pred = func.predict(model, x_test=test_data[key][0])

            d = {'Column': key}
            metric_dict = metric.all_metric(y=test_data[key][1], pred=pred.reshape(-1))
            d.update(metric_dict)
            result_list.append(d)

        # 写到 CSV 文件里
        csv_name = func.__name__.split('.')[-1] + f'_{gen_dataset.future_days}day' + '_' + filename.split('/')[-1]
        data_process.dump_csv(res_dir1, csv_name, result_list, avg=avg)


def supervised_model(func, data, filename):
    print('模型：', func.__name__, 'future_days:', gen_dataset.future_days)

    result_list = []
    for column in data:
        pred = func(x_train=data[column][0], y_train=data[column][1], x_test=data[column][2])

        d = {'Column': column}
        metric_dict = metric.all_metric(y=data[column][3], pred=pred.reshape(-1))
        d.update(metric_dict)
        result_list.append(d)

    # 写入 CSV 文件
    csv_name = func.__name__.split('_')[0] + f'_{gen_dataset.future_days}day' + '_' + filename.split('/')[-1]
    data_process.dump_csv(res_dir2, csv_name, result_list, avg=avg)


def predict_all_data(func, filename):
    x_train, y_train, x_test, y_test = gen_dataset.load_all(filename)

    pred = func(x_train, y_train, x_test)
    print(func.__name__, filename, gen_dataset.future_days)
    print(metric.all_metric(y_test, pred))


def predict_cols(filename):
    # 方案 2

    data = gen_dataset.load_cols(filename)

    for func in [lr.lr_predict, svr.svr_predict, xgb.xgb_predict, mlp.mlp_predict]:
        supervised_model(func, data, filename)


def avg(series):
    series = series.dropna()
    return sum(series) / len(series)


if __name__ == '__main__':
    pred_target = get_config('config.json', 'predict-target')
    pred_target_filename = get_config('../data/data.json', pred_target, 'server')

    # predict_cols(strainmeter)
    # predict_every_col(strainmeter)
    target_data = gen_dataset.load_cols(pred_target_filename)
    supervised_model(lstm.lstm_union_predict, target_data, pred_target_filename)
    # predict_all_data(lstm.lstm_union_predict, taseometer)
