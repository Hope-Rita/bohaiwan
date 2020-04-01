import json


class Config(object):

    def __init__(self, runtime_config_path, data_config_path='../data/data.json'):
        """
        初始化全局的运行配置和数据存储配置
        :param runtime_config_path: 存放运行配置的路径
        :param data_config_path: 存放数据存储配置的路径
        """
        # 加载程序运行配置
        f = open(runtime_config_path, encoding='utf-8')
        self._config_dict = json.load(f)
        f.close()
        # 加载数据存储配置
        f = open(data_config_path, encoding='utf-8')
        self._data_config_dict = json.load(f)
        f.close()

    def modify_config(self, *keys, new_val):
        """
        修改配置
        :param keys: 逐层的关键字
        :param new_val: 新赋的值
        """
        # 逐层取值
        d = self._config_dict
        for key in keys[:-1]:
            d = d[key]

        # 修改，赋新值
        d[keys[-1]] = new_val

    def get_config(self, *keys, inner_keys=None):
        """
        从 JSON 文件中加载配置信息
        :param keys: 从浅到深的关键字信息
        :param inner_keys: 最内层的关键字列表，在需要加载多个值的时候使用
        :return: 加载单个值时只返回一个值，加载多个值时返回元组
        """
        # 检查参数是否合法
        if inner_keys and type(inner_keys) is not list:
            raise TypeError('inner_keys 应为一个列表')

        # 逐层取值
        res = self._config_dict
        for key in keys:
            res = res[key]

        if not inner_keys:
            return res
        else:
            return tuple([res[inner_key] for inner_key in inner_keys])

    def get_data_loc(self, data_name, local=False):
        """
        获取存放数据的路径
        :param data_name: 数据的名字
        :param local: 若为 True 则加载本地的文件，否则加载服务器上的
        :return: 文件的绝对路径
        """
        return self._data_config_dict[data_name]['local' if local else 'server']


config_path = '../union_predict/pred_len_survey.json'
global_config = Config(config_path)
