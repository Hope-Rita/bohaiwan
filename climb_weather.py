import datetime
import pandas as pd
import re
import requests
import time
from bs4 import BeautifulSoup

headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36'
}


def get_html(url):
    while True:
        r = requests.get(url, headers=headers)
        print('从', url, '获取数据')
        if 'table' in r.text:
            print('成功获取数据')
            return r.content
        else:
            print('请求失败，重试')
            time.sleep(1)


def parse_html(page_content):
    soup = BeautifulSoup(page_content, features='lxml')
    table = soup.find('table')
    item_list = table.find_all('tr')

    month = []

    for i in range(1, len(item_list)):
        td = item_list[i].find_all('td')
        day = list()

        # 日期
        day.append(parse_date(td[0].a.getText()))

        # 高温低温
        nums = re.findall(r'-?\d+', td[2].getText())
        day.append(int(nums[1]))
        day.append(int(nums[0]))

        # 天气和风向
        pattern = re.compile(r'\s+')
        day.append(re.sub(pattern, '', td[1].getText()))
        day.append(re.sub(pattern, '', td[3].getText()))

        month.append(day)

    return month


def parse_date(text):
    y = text.find('年')
    m = text.find('月')
    d = text.find('日')
    return datetime.date(int(text[y - 4: y]), int(text[m - 2: m]), int(text[d - 2: d]))


def main():

    data = []
    for year in [2016, 2017]:
        for month in range(1, 13):
            print(f'爬取{year}年{month}月的天气数据')
            month_str = '0' + str(month) if month < 10 else str(month)
            url = 'http://www.tianqihoubao.com/lishi/nanjing/month/' + str(year) + month_str + '.html'

            h = get_html(url)
            data.extend(parse_html(h))

    frame = pd.DataFrame(data, columns=['date', 'low_tp', 'high_tp', 'weather', 'wind'])
    frame.to_csv('weather.csv', index=False)


if __name__ == '__main__':
    main()
