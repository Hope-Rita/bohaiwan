import torch
from torch import nn
from utils.config import *


# 加载模型参数
rnn_hidden_size = global_config.get_config('model-parameters', 'recurrent', 'rnn-hidden-size')
gru_hidden_size = global_config.get_config('model-parameters', 'recurrent', 'gru-hidden-size')
lstm_hidden_size = global_config.get_config('model-parameters', 'recurrent', 'lstm-hidden-size')

# 加载数据参数
pred_len, env_factor_num = global_config.get_config('data-parameters', inner_keys=['pred-len', 'env-factor-num'])


class FusionBase(nn.Module):

    def __init__(self, time_series_len, hidden_size, output_size=1):
        """
        :param time_series_len: 时间序列的长度
        :param hidden_size: 隐单元数量
        :param output_size: 输出规模
        """
        super(FusionBase, self).__init__()
        self.fc_fusion = nn.Linear(hidden_size + env_factor_num, output_size)
        self.time_series_len = time_series_len

    def forward(self, rnn_output, env_factor_vec):
        s, b, h = rnn_output.shape
        rnn_output = rnn_output.view(s * b, h)
        env_factor_vec = env_factor_vec.view(env_factor_vec.shape[0] * env_factor_vec.shape[1], -1)

        x = torch.cat((rnn_output, env_factor_vec), dim=1)
        res = self.fc_fusion(x)
        res = res.view(-1)
        return res


class RNNFusion(FusionBase):

    def __init__(self, time_series_len, input_feature, hidden_size=rnn_hidden_size, output_size=1):
        super(RNNFusion, self).__init__(time_series_len, hidden_size, output_size)
        self.rnn = nn.RNN(input_size=input_feature, hidden_size=hidden_size)

    def forward(self, input_x, _=None):
        x = input_x[:, :, :self.time_series_len]
        x = x.permute(2, 0, 1)
        e = input_x[:, :, self.time_series_len:]
        output, hn = self.rnn(x)
        return super().forward(hn, e)


class GRUFusion(FusionBase):

    def __init__(self, time_series_len, input_feature, hidden_size=gru_hidden_size, output_size=1):
        super(GRUFusion, self).__init__(time_series_len, hidden_size, output_size)
        self.gru = nn.GRU(input_size=input_feature, hidden_size=hidden_size)

    def forward(self, input_x, _=None):
        x = input_x[:, :, :self.time_series_len]
        x = x.permute(2, 0, 1)
        e = input_x[:, :, self.time_series_len:]
        output, (hn, cn) = self.gru(x)
        return super(GRUFusion, self).forward(hn, e)


class LSTMFusion(FusionBase):

    def __init__(self, time_series_len, input_feature, hidden_size=lstm_hidden_size, output_size=1):
        super(LSTMFusion, self).__init__(time_series_len, hidden_size, output_size)
        self.lstm = nn.LSTM(input_size=input_feature, hidden_size=hidden_size)

    def forward(self, input_x, _=None):
        x = input_x[:, :, :self.time_series_len]
        x = x.permute(2, 0, 1)  # LSTM 的输入为 (seq_len, batch, input_size)
        e = input_x[:, :, self.time_series_len:]
        output, (hn, cn) = self.lstm(x)
        return super().forward(hn, e)
