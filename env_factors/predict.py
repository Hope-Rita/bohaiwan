"""
采取三种方案来预测
1）所有的数据一起训练和测试
2）所有的数据一起训练，分列进行测试
3）分列进行训练和测试
"""


from baseline import lr
from baseline import mlp
from baseline import svr
from baseline import xgb
from env_factors import gen_dataset
from utils import data_process
from utils import metric
from utils.config import get_config


# 存放预测结果的文件路径
result_dir, result2_dir = get_config('config.json', 'predict-result', inner_keys=['result', 'result2'])


def supervised_model(filename, func, one_col=False):

    if one_col:
        x_train, y_train, x_test, y_test, normal_y = gen_dataset.load_one_col(filename, col)
    else:
        x_train, y_train, x_test, y_test, normal_y = gen_dataset.load_all(filename)

    pred = func(x_train, y_train, x_test)

    # 反归一化
    pred = normal_y.inverse_transform(pred)
    y_test = normal_y.inverse_transform(y_test)

    print(func.__name__, filename)
    print(metric.all_metric(y_test, pred))


def all_supervised_models(filename):
    # x_train, y_train, x_test, y_test, normal_y = gen_dataset.load_all(filename)
    x_train, y_train, x_test, y_test = gen_dataset.load_all(filename)
    for func in [lr.lr_predict, svr.svr_predict, xgb.xgb_predict, mlp.mlp_predict]:
        pred = func(x_train, y_train, x_test)

        # 反归一化
        # pred = normal_y.inverse_transform(pred)
        # y_test = normal_y.inverse_transform(y_test)

        print(func.__name__, filename)
        print(metric.all_metric(y_test, pred))


def predict_every_col(filename):
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
        csv_name = func.__name__.split('.')[-1] + '_' + filename.split('\\')[-1]
        data_process.dump_csv(result_dir, csv_name, result_list)
        print('完成预测，已写入', csv_name)


def predict_one_cols(filename):
    data = gen_dataset.load_one_cols(filename)

    for func in [lr, svr, xgb, mlp]:
        print('模型：', func.__name__)

        result_list = []

        for key in data:
            pred = direct_predict(func, x_train=data[key][0], y_train=data[key][1], x_test=data[key][2])

            d = {'Column': key}
            metric_dict = metric.all_metric(y=data[key][3], pred=pred.reshape(-1))
            d.update(metric_dict)
            result_list.append(d)

        # 写到 CSV 文件里
        csv_name = func.__name__.split('.')[-1] + '_' + filename.split('/')[-1]
        data_process.dump_csv(result2_dir, csv_name, result_list)
        print('完成预测，已写入', csv_name)


def direct_predict(func, x_train, y_train, x_test):
    model = func.model_fit(x_train, y_train)
    return model.predict(x_test)


if __name__ == '__main__':
    pred_target = get_config('config.json', 'predict-target')
    pred_target_filename = get_config('../data/data.json', pred_target, 'server')
    predict_one_cols(pred_target_filename)
