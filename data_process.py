import datetime
import numpy as np
import pandas as pd


name1 = 'data\\strainmeter.csv'
name2 = 'data\\pressure.csv'
name3 = 'data\\temperature.csv'
name4 = 'data\\taseometer.csv'
name5 = 'data\\waterline.csv'


def process_data(frame, diff=False):

    for i in range(len(frame)):
        for j in range(1, len(frame.columns)):
            if frame.iloc[i, j] == 0:
                frame.iloc[i, j] = np.nan

    frame.fillna(method='ffill', inplace=True)

    if diff:
        for i in range(1, len(frame.columns)):
            horizon = frame.iloc[0, i]
            for j in range(len(frame)):
                frame.iloc[j, i] -= horizon

    return frame


def date_format(frame):
    for i in range(len(frame)):
        d = frame.loc[i, 'date']
        s = d.split('/')

        frame.loc[i, 'date'] = datetime.date(int(s[0]), int(s[1]), int(s[2]))

    return frame


f = pd.read_csv(name2)
f = process_data(f, True)
f.to_csv(name2, index=False)
