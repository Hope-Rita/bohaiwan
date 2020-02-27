import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


def all_predict(y_dict):

    for key in y_dict:
        y = y_dict[key]
        if key != 'True value':
            tmp = np.array([np.nan] * (len(y_dict['True value']) - len(y)))
            tmp[-1] = y_dict['True value'][len(y_dict['True value']) - len(y)]
            y = np.concatenate((tmp, y))

        plt.plot(y[200:], label=key)

    plt.legend()
    plt.show()


def metric_for_cols():
    result1_dir = '/home/lwt/bohaiwan/union_predict/result1'
    result1_dir = 'C:\\study\\WorkSpace\\bohaiwan\\union_predict\\result1'

    file_list = os.listdir(result1_dir)
    valid_files = [f for f in file_list if '7day_s' in f]

    for f in valid_files:
        frame = pd.read_csv(result1_dir + '\\' + f)
        label = f.split('_')[0].split('.')[1]
        plt.plot(frame.loc[:, 'MAE'].to_numpy(), label=label)

    plt.legend()
    plt.show()
    print('ssss')


if __name__ == '__main__':
    metric_for_cols()
