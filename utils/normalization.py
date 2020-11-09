import numpy as np


np.random.seed(1337)  # for reproducibility


class MinMaxNormal(object):
    """
        MinMax Normalization -> [0, 1]
        x = (x - min) / (max - min).
    """
    def __init__(self, x):

        if type(x) is list:
            self._min = min([item.min() for item in x])
            self._max = max([item.max() for item in x])
        else:
            self._min = x.min()
            self._max = x.max()

    def transform(self, x):
        x = 1.0 * (x - self._min) / (self._max - self._min)
        return x

    def inverse_transform(self, x):
        x = 1.0 * x * (self._max - self._min) + self._min
        return x

    def span(self):
        return self._max - self._min


class MinMaxNormal2(MinMaxNormal):
    """
        MinMax Normalization -> [-1, 1]
        x = (x - min) / (max - min).
        x = x * 2 - 1
    """

    def __init__(self, x):
        super(MinMaxNormal2, self).__init__(x)

    def transform(self, x):
        x = 1.0 * (x - self._min) / (self._max - self._min)
        x = x * 2.0 - 1.0
        return x

    def inverse_transform(self, x):
        x = (x + 1.0) / 2.0
        x = 1.0 * x * (self._max - self._min) + self._min
        return x


def sigmoid(data):
    return 1.0 / (1 + np.exp(-data))
