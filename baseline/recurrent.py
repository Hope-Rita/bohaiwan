import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange

from utils.config import Config
from baseline.recurrent_reg import RNNReg, GRUReg, LSTMReg
from baseline.recurrent_fusion import RNNFusion, GRUFusion, LSTMFusion
from baseline.recurrent_section_fusion import LSTMSectionFusion
from utils import normalization
from utils.metric import RMSELoss


conf = Config()
# 加载模型参数
device = torch.device(conf.get_config('device', 'cuda') if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
num_workers, batch_size, epoch_num, learning_rate \
            = conf.get_config('model-parameters',
                               'recurrent',
                               inner_keys=['num-workers', 'batch-size', 'epoch-num', 'learning-rate']
                               )
rnn_hidden_size = conf.get_config('model-parameters', 'recurrent', 'rnn-hidden-size')
gru_hidden_size = conf.get_config('model-parameters', 'recurrent', 'gru-hidden-size')
lstm_hidden_size = conf.get_config('model-parameters', 'recurrent', 'lstm-hidden-size')

# 加载数据参数
pred_len, env_factor_num = conf.get_config('data-parameters', inner_keys=['pred-len', 'env-factor-num'])


def union_predict(model, x_train, y_train, x_test):
    # 加载数据
    # data_loader, x_test, normal = get_dataloader(x_train, y_train, x_test, normalize=True)
    data_loader, x_test = get_dataloader(x_train, y_train, x_test, normalize=False)

    # 训练模型
    model = train_model(model, data_loader)

    # 将输出的结果进行处理并返回
    pred = model(x_test)
    pred = pred.data.to('cpu').numpy()
    pred = pred.reshape(-1)
    # pred = normal.inverse_transform(pred)
    return pred


def lstm_union_predict(x_train, y_train, x_test):
    model = LSTMFusion(time_series_len=pred_len, input_feature=1).to(device)
    return union_predict(model, x_train, y_train, x_test)


def gru_union_predict(x_train, y_train, x_test):
    model = GRUFusion(time_series_len=pred_len, input_feature=1).to(device)
    return union_predict(model, x_train, y_train, x_test)


def rnn_union_predict(x_train, y_train, x_test):
    model = RNNFusion(time_series_len=pred_len, input_feature=1).to(device)
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

    with trange(epoch_num) as t:

        for epoch in t:

            t.set_description(f'[epoch: {epoch}, lr:{learning_rate}]')
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
            t.set_postfix(min_loss=min_loss, min_epoch=min_epoch)

            scheduler.step(loss)  # 更新学习率

    return model


def get_dataloader(x_train, y_train, x_test, normalize=True):
    normal = None
    if normalize:  # 归一化
        normal = normalization.MinMaxNormal(y_train)
        # x_train = normal.transform(x_train)
        y_train = normal.transform(y_train)
        # x_test = normal.transform(x_test)

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
