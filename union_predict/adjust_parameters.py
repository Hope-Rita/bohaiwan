from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from utils import metric
from utils.config import *
from union_predict import gen_dataset
from utils import normalization
from utils import data_process


def search_parameters(func, parameters, data):
    x_train, y_train = data
    grid_search = GridSearchCV(func,
                               parameters,
                               scoring=make_scorer(metric.rmse, greater_is_better=False),
                               cv=5
                               )
    grid_search.fit(x_train, y_train)
    return grid_search


def search_all_cols(func, parameters, model_name):

    pred_target = global_config.get_config('predict-target')
    pred_target_filename = global_config.get_data_loc(pred_target)
    target_data = gen_dataset.load_cols(pred_target_filename, random_pick=False)

    para_list = []  # 存放各个列的最佳参数
    for col in target_data:
        print(f'对 {col} 进行调参')
        col_best_para = {'col_name': col}
        x_train, y_train, x_test, y_test = target_data[col]

        # 正则化
        normal = normalization.MinMaxNormal([x_train, y_train, x_test, y_test])
        x_train, y_train, x_test, y_test \
            = normal.transform(x_train), normal.transform(y_train), normal.transform(x_test), normal.transform(y_test)

        grid_search = search_parameters(func, parameters, (x_train, y_train))
        col_best_para.update(grid_search.best_params_)
        para_list.append(col_best_para)
        print('Test set score', grid_search.score(x_test, y_test))
        print("Best parameters:{}".format(grid_search.best_params_))
        print("Best score on train set:{:.2f}".format(grid_search.best_score_))

    # 把结果写入 CSV 文件中
    data_process.dump_csv('adjust_para', f'{model_name}.csv', para_list)


def xgb():
    from xgboost import XGBRegressor

    paras = {
        'max_depth': [depth for depth in range(1, 9)],
        'learning_rate': [0.03, 0.1, 0.3, 1],
        'n_estimators': [5, 10, 15, 20, 25, 30],
        'gamma': [0, 1, 2, 3]
    }

    test_func = XGBRegressor(objective='reg:squarederror')
    search_all_cols(test_func, paras, 'xgb')


if __name__ == '__main__':
    xgb()