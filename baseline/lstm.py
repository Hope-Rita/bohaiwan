import numpy as np
import time
import torch
from baseline import normalization
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from union_predict import gen_dataset
from utils.config import get_config


config_path = 'config.json'
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


class LstmReg(nn.Module):

    def __init__(self, input_size, hidden_size=model_hidden_size, output_size=1, num_layers=2):
        super(LstmReg, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.reg = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.lstm(x)
        s, b, h = x.shape
        x = x.view(s * b, h)
        x = self.reg(x)
        x = x.view(s, b, -1)
        return x


class FusionLstm(nn.Module):

    def __init__(self, time_series_len, hidden_size=model_hidden_size, output_size=1, num_layers=2):
        super(FusionLstm, self).__init__()
        self.lstm = nn.LSTM(time_series_len, hidden_size, num_layers)
        self.fc_fusion = nn.Linear(hidden_size + gen_dataset.env_factor_num, output_size)
        self.time_series_len = time_series_len

    def forward(self, input_x):
        x = input_x[:, :, :self.time_series_len]
        e = input_x[:, :, self.time_series_len:]
        x, _ = self.lstm(x)

        s, b, h = x.shape
        x = x.view(s * b, h)
        e = e.view(e.shape[0] * e.shape[1], -1)

        x = torch.cat((x, e), dim=1)
        x = self.fc_fusion(x)
        x = x.view(s, b, -1)
        return x


def lstm_union_predict(x_train, y_train, x_test):
    data_loader, x_test = get_dataloader(x_train, y_train, x_test, normalize=False)
    model = FusionLstm(time_series_len=gen_dataset.pred_len).to(device)
    model = train_model(model, data_loader)

    pred = model(x_test)
    pred = pred.data.to('cpu').numpy()
    pred = pred.reshape(-1)
    return pred


def lstm_predict(x_train, y_train, x_test):
    # 输入的数据格式为 numpy 数组

    data_loader, x_test, normal = get_dataloader(x_train, y_train, x_test)

    # 训练模型
    model = LstmReg(x_train.shape[-1]).to(device)
    model = train_model(model, data_loader)

    pred = model(x_test)
    pred = pred.data.to('cpu').numpy()
    pred = pred.reshape(-1)
    return normal.inverse_transform(pred)


def train_model(model, data_loader):
    mse = nn.MSELoss()
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
                loss = mse(pred_y, y)
                train_loss += loss.item()

                opt.zero_grad()
                loss.backward()
                opt.step()

            scheduler.step(loss)

        train_loss /= data_loader.batch_size
        if train_loss < min_loss:
            min_loss = train_loss
            min_epoch = epoch
        print(f'min_loss:{min_loss}, min_epoch:{min_epoch}')

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
    y_train = y_train.reshape(-1, 1, 1)
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
