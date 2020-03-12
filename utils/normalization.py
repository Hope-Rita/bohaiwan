import numpy as np

np.random.seed(1337)  # for reproducibility


class MinMaxNormal(object):
    """
        MinMax Normalization --> [-1, 1]
        x = (x - min) / (max - min).
        x = x * 2 - 1
    """
    def __init__(self, x):

        if type(x) is list:
            self._min = min([item.min() for item in x])
            self._max = max([item.max() for item in x])
        else:
            self._min = x.min()
            self._max = x.max()

    def transform(self, x):
        x = 1. * (x - self._min) / (self._max - self._min)
        x = x * 2. - 1.
        return x

    def inverse_transform(self, x):
        x = (x + 1.) / 2.
        x = 1. * x * (self._max - self._min) + self._min
        return x

    def span(self):
        return self._max-self._min
