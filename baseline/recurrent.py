import numpy as np
import time
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from utils import normalization
from utils.config import get_config
from utils.metric import RMSELoss
from union_predict.gen_dataset import pred_len, env_factor_num


config_path = '../union_predict/config.json'
# 参数加载
device = torch.device(get_config(config_path, 'device', 'cuda') if torch.cuda.is_available() else 'cpu')
model_hidden_size, num_workers, batch_size, epoch_num, learning_rate \
    = get_config(config_path,
                 'model-parameters',
                 'lstm',
                 inner_keys=['model-hidden-size',
                             'num-workers',
                             'batch-size',
                             'epoch-num',
                             'learning-rate'
                             ]
                 )
rnn_hidden_size = get_config(config_path, 'model-parameters', 'lstm', 'rnn-hidden-size')
gru_hidden_size = get_config(config_path, 'model-parameters', 'lstm', 'gru-hidden-size')


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

    def __init__(self, input_size, hidden_size=model_hidden_size, output_size=1):
        super(LSTMReg, self).__init__(hidden_size, output_size)
        self.lstm = nn.LSTM(input_size, hidden_size)

    def forward(self, x):
        x, _ = self.lstm(x.permute(2, 0, 1))
        return super().forward(x)


class FusionBase(nn.Module):

    def __init__(self, time_series_len, input_features, hidden_size, output_size=1):
        """
        :param time_series_len: 时间序列的长度
        :param input_features: 每个时间点元素的特征数
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

    def __init__(self, time_series_len, hidden_size=rnn_hidden_size, output_size=1):
        super(RNNFusion, self).__init__(time_series_len, hidden_size, output_size)
        self.rnn = nn.RNN(time_series_len, hidden_size)

    def forward(self, input_x, _=None):
        x = input_x[:, :, :self.time_series_len]
        e = input_x[:, :, self.time_series_len:]
        x, _ = self.rnn(x.permute(2, 0, 1))

        return super().forward(x, e)


class GRUFusion(FusionBase):

    def __init__(self, time_series_len, hidden_size=gru_hidden_size, output_size=1):
        super(GRUFusion, self).__init__(time_series_len, hidden_size, output_size)
        self.gru = nn.GRU(time_series_len, hidden_size)

    def forward(self, input_x, _=None):
        x = input_x[:, :, :self.time_series_len]
        e = input_x[:, :, self.time_series_len:]
        x, _ = self.gru(x.permute(2, 0, 1))

        return super(GRUFusion, self).forward(x, e)


class LSTMFusion(FusionBase):

    def __init__(self, time_series_len, input_feature, hidden_size=model_hidden_size, output_size=1):
        super(LSTMFusion, self).__init__(time_series_len, input_feature, hidden_size, output_size)
        self.lstm = nn.LSTM(input_size=input_feature, hidden_size=hidden_size)

    def forward(self, input_x, _=None):
        x = input_x[:, :, :self.time_series_len]
        x = x.permute(2, 0, 1)  # LSTM 的输入为 (seq_len, batch, input_size)
        e = input_x[:, :, self.time_series_len:]
        output, (hn, cn) = self.lstm(x)
        return super().forward(hn, e)


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

    def __init__(self, seq_len, time_series_len, env_factor_len, hidden_size=model_hidden_size, output_size=1):
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


def union_predict(model, x_train, y_train, x_test):
    data_loader, x_test, normal = get_dataloader(x_train, y_train, x_test, normalize=True)

    model = train_model(model, data_loader)
    pred = model(x_test)
    pred = pred.data.to('cpu').numpy()
    pred = pred.reshape(-1)
    pred = normal.inverse_transform(pred)
    return pred


def lstm_union_predict(x_train, y_train, x_test):
    model = LSTMFusion(time_series_len=pred_len, input_feature=1).to(device)
    return union_predict(model, x_train, y_train, x_test)


def gru_union_predict(x_train, y_train, x_test):
    model = GRUFusion(time_series_len=pred_len).to(device)
    return union_predict(model, x_train, y_train, x_test)


def rnn_union_predict(x_train, y_train, x_test):
    model = RNNFusion(time_series_len=pred_len).to(device)
    return union_predict(model, x_train, y_train, x_test)


def lstm_section_predict(x_train, y_train, x_test):
    print(x_train.shape)
    model = LSTMSectionFusion(seq_len=x_train.shape[1], time_series_len=pred_len, env_factor_len=env_factor_num)
    return union_predict(model, x_train, y_train, x_test)


def lstm_predict(x_train, y_train, x_test):
    # 输入的数据格式为 numpy 数组
    data_loader, x_test, normal = get_dataloader(x_train, y_train, x_test)

    # 训练模型
    model = LSTMReg(x_train.shape[-1]).to(device)
    model = train_model(model, data_loader)

    pred = model(x_test)
    pred = pred.data.to('cpu').numpy()
    pred = pred.reshape(-1)
    return normal.inverse_transform(pred)


def train_model(model, data_loader):
    rmse = RMSELoss()
    opt = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2, threshold=1e-3, min_lr=1e-6)
    min_loss = np.inf
    min_epoch = 0

    start_time = time.time()
    for epoch in range(epoch_num):

        print(f'[epoch: {epoch}], lr={learning_rate}, time usage: {int(time.time() - start_time)}s')
        model.train()

        train_loss = 0.0
        for i, data in enumerate(data_loader):
            x, y = data

            with torch.set_grad_enabled(True):
                pred_y = model(x)
                loss = rmse(pred_y, y)
                train_loss += loss.item() * len(x)

                opt.zero_grad()
                loss.backward()
                opt.step()

        train_loss /= len(data_loader.dataset)
        if train_loss < min_loss:
            min_loss = train_loss
            min_epoch = epoch
        print(f'min_loss:{min_loss}, min_epoch:{min_epoch}')

        scheduler.step(loss)  # 更新学习率

    return model


def get_dataloader(x_train, y_train, x_test, normalize=True):
    normal = None
    if normalize:  # 归一化
        normal = normalization.MinMaxNormal([x_train, y_train, x_test])
        x_train = normal.transform(x_train)
        y_train = normal.transform(y_train)
        x_test = normal.transform(x_test)

    # 改变形状格式
    x_train = x_train.reshape(-1, 1, x_train.shape[1])
    x_test = x_test.reshape(-1, 1, x_test.shape[1])

    # 转换成 tensor
    x_train = torch.from_numpy(x_train).float().to(device)
    y_train = torch.from_numpy(y_train).float().to(device)
    x_test = torch.from_numpy(x_test).float().to(device)

    # 构建 DataSet 和 DataLoader
    dataset = TensorDataset(x_train, y_train)
    data_loader = DataLoader(dataset=dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers)

    if normalize:
        return data_loader, x_test, normal
    else:
        return data_loader, x_test
