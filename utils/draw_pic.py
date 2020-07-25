import matplotlib.pyplot as plt
import numpy as np
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


def compare(y, pred, save_path, title_info=None):
    plt.figure(figsize=(15, 8))
    plt.plot(pred, color='green', label='predict')
    plt.plot(y, label='truth')

    if title_info:
        plt.title(title_info)

    plt.legend()

    # 保存图像, 如果路径不存在则要新建
    if not os.path.exists(save_path['dir']):
        os.makedirs(save_path['dir'])
    plt.savefig(os.path.join(save_path['dir'], save_path['filename']))

    plt.clf()
    plt.close('all')


def draw_by_label(pic_path, pic_name, **items):
    plt.figure()
    for label in items:
        plt.plot(items[label], label=label)

    plt.title(pic_name)
    plt.legend()

    if not os.path.exists(pic_path):
        os.makedirs(pic_path)
    plt.savefig(os.path.join(pic_path, pic_name))

    plt.clf()
    plt.close('all')