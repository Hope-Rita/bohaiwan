import json


def modify_config(json_path, *keys, new_val):
    """
    修改配置文件
    :param json_path: JSON 文件的路径
    :param keys: 逐层的关键字
    :param new_val: 新赋的值
    """
    # 加载 JSON 字典
    f = open(json_path, encoding='utf-8')
    config_dict = json.load(f)
    f.close()

    # 逐层取值
    d = config_dict
    for key in keys[:-1]:
        d = d[key]

    # 修改，赋新值
    d[keys[-1]] = new_val

    # 写入 JSON 文件
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=4, separators=(',', ':'))


def get_config(json_path, *keys, inner_keys=None):
    """
    从 JSON 文件中加载配置信息
    :param json_path: JSON 文件路径
    :param keys: 从浅到深的关键字信息
    :param inner_keys: 最内层的关键字列表，在需要加载多个值的时候使用
    :return: 加载单个值时只返回一个值，加载多个值时返回元组
    """
    # 检查参数是否合法
    if inner_keys and type(inner_keys) is not list:
        raise TypeError('inner_keys 应为一个列表')

    # 加载 JSON 字典
    f = open(json_path, encoding='utf-8')
    config_dict = json.load(f)
    f.close()

    # 逐层取值
    res = config_dict
    for key in keys:
        res = res[key]

    if not inner_keys:
        return res
    else:
        return tuple([res[inner_key] for inner_key in inner_keys])
