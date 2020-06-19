import torch
from torch import nn
from utils.config import Config


conf = Config()
# 加载模型参数
rnn_hidden_size = conf.get_config('model-parameters', 'recurrent', 'rnn-hidden-size')
gru_hidden_size = conf.get_config('model-parameters', 'recurrent', 'gru-hidden-size')
lstm_hidden_size = conf.get_config('model-parameters', 'recurrent', 'lstm-hidden-size')

# 加载数据参数
pred_len, env_factor_num = conf.get_config('data-parameters', inner_keys=['pred-len', 'env-factor-num'])


class SectionFusion(nn.Module):

    def __init__(self, time_series_len, env_factor_len, fc_hidden_size, output_size=1):
        super(SectionFusion, self).__init__()
        self.fc_fusion = nn.Linear(fc_hidden_size, output_size)
        self.time_series_len = time_series_len
        self.env_factor_len = env_factor_len

    def forward(self, rnn_output, env_factor_vec):
        s, b, h, c = rnn_output.shape
        rnn_output = rnn_output.view(s * b, h * c)
        env_factor_vec = env_factor_vec.view(env_factor_vec.shape[0] * env_factor_vec.shape[1], -1)

        x = torch.cat((rnn_output, env_factor_vec), dim=1)
        res = self.fc_fusion(x)
        res = res.view(s, b, -1)
        return res


class LSTMSectionFusion(SectionFusion):

    def __init__(self, seq_len, time_series_len, env_factor_len, hidden_size=lstm_hidden_size, output_size=1):
        super(LSTMSectionFusion, self).__init__(time_series_len, env_factor_len, seq_len - env_factor_len)
        self.lstm = nn.LSTM(time_series_len, hidden_size)

    def forward(self, input_x, _=None):
        x = input_x[:, :, :-self.env_factor_len]
        e = input_x[:, :, -self.env_factor_len:]

        s, b, h = x.shape
        print(x[0][0])
        x = x.view(s, b, self.time_series_len, h // self.time_series_len)
        print(x[0][0])
        print(x.shape)
        print(e.shape)
        x, _ = self.lstm(x)
        return super().forward(x, e)
