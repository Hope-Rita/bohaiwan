import json
import pandas as pd
from utils.config import get_config


data_path = '../data/data.json'
json_path = get_config('config.json', 'col-match', 'server')
taseometer = get_config(data_path, 'taseometer', 'server')
strainmeter = get_config(data_path, 'strainneter', 'server')
pressure = get_config(data_path, 'pressure', 'server')
weather = get_config(data_path, 'weather', 'server')
waterline = get_config(data_path, 'waterline', 'server')
temperature = get_config(data_path, 'temperature', 'server')


pressure_frame = pd.read_csv(pressure, parse_dates=True, index_col='date')
temperature_frame = pd.read_csv(temperature, parse_dates=True, index_col='date')


def cal_target_cols(col):
    return {'pressure': match(col, pressure_frame), 'temperature': match(col, temperature_frame)}


def match(col, frame):
    """
    给 col 匹配对应的列
    :param col: 表示列号的字符串
    :param frame: 待匹配的 DataFrame
    :return: List of matched cols
    """
    # 每个 col 形式为 'Item_Section_SubSection_Index'
    s = col.split('_')
    section = s[1]
    subsection = s[2]
    index = s[3]

    target_cols = frame.columns[1:]

    # 先挑出 Section 一样的
    valid_sections = []
    for c in target_cols:
        if c.split('_')[1] == section:
            valid_sections.append(c)

    # 1：最优匹配 Section_SubSection_Index 一样的
    for c in valid_sections:
        s = c.split('_')
        if s[2] == subsection and s[3] == index:
            return [c]

    # 2: 匹配 SubSection 一样的
    res = []
    for c in valid_sections:
        s = c.split('_')
        if s[2] == subsection:
            res.append(c)

    if res:
        return res

    # 3: 匹配 SubSection 相近的 (即 SubSection 的第一个字母一样的)
    res = []
    for c in valid_sections:
        s = c.split('_')
        if s[2][0] == subsection[0]:
            res.append(c)

    if res:
        return res

    # 4: 都找不到就返回同一个 Section 的
    return valid_sections


def main():
    d = {}

    frame = pd.read_csv(taseometer)
    cols = frame.columns[1:]
    for col in cols:
        d[col] = cal_target_cols(col)

    frame = pd.read_csv(strainmeter)
    cols = frame.columns[1:]
    for col in cols:
        d[col] = cal_target_cols(col)

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(d, f, ensure_ascii=False, sort_keys=True, indent=4, separators=(',', ':'))


if __name__ == '__main__':
    main()
