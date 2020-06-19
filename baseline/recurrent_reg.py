from torch import nn
from utils.config import Config


conf = Config()
# 加载模型参数
rnn_hidden_size = conf.get_config('model-parameters', 'recurrent', 'rnn-hidden-size')
gru_hidden_size = conf.get_config('model-parameters', 'recurrent', 'gru-hidden-size')
lstm_hidden_size = conf.get_config('model-parameters', 'recurrent', 'lstm-hidden-size')

# 加载数据参数
pred_len, env_factor_num = conf.get_config('data-parameters', inner_keys=['pred-len', 'env-factor-num'])


class RegBase(nn.Module):

    def __init__(self, hidden_size, output_size=1):
        super(RegBase, self).__init__()
        self.reg = nn.Linear(hidden_size, output_size)

    def forward(self, rnn_output):
        s, b, h = rnn_output.shape
        rnn_output = rnn_output.view(s * b, h)
        res = self.reg(rnn_output)
        res = res.view(s, b, -1)
        return res


class RNNReg(RegBase):

    def __init__(self, input_size, hidden_size=rnn_hidden_size, output_size=1):
        super(RNNReg, self).__init__(hidden_size, output_size)
        self.rnn = nn.RNN(input_size, hidden_size)

    def forward(self, x):
        x, _ = self.rnn(x.permute(2, 0, 1))
        return super().forward(x)


class GRUReg(RegBase):

    def __init__(self, input_size, hidden_size=gru_hidden_size, output_size=1):
        super(GRUReg, self).__init__(hidden_size, output_size)
        self.gru = nn.GRU(input_size, hidden_size)

    def forward(self, x):
        x, _ = self.gru(x.permute(2, 0, 1))
        return super().forward(x)


class LSTMReg(RegBase):

    def __init__(self, input_size, hidden_size=lstm_hidden_size, output_size=1):
        super(LSTMReg, self).__init__(hidden_size, output_size)
        self.lstm = nn.LSTM(input_size, hidden_size)

    def forward(self, x):
        x, _ = self.lstm(x.permute(2, 0, 1))
        return super().forward(x)
