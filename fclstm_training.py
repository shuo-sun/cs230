import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from load_data import *
import global_consts as cnst
from fclstm import LSTMForecast, LSTMAQI

device = torch.device('cuda')


def get_seqs():
    grid_datas, aqi_datas = load_batch_seq_data()

    return grid_datas[0], aqi_datas[0]


def train_fclstm(grid_seqs, aqi_seqs, hidden_size=1024, learning_rate=0.01):
    model = LSTMForecast(hidden_size, 10).cuda()
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), learning_rate)

    for epoch in range(100):
        for i in range(len(grid_seqs) - 1):
            grid_seq = grid_seqs[i]
            aqi_seq = aqi_seqs[i]

            input_data = torch.tensor(
                grid_seq[:, :-1, :, :, :],  # Remove the last in the seq
                dtype=torch.float32,
                device=device,
            )
            target = torch.tensor(
                np.concatenate((
                    np.reshape(grid_seq[:, 1:, :5, :, :], (10, 71, -1)),
                    aqi_seq[:, 1:, :]),
                    axis=-1),
                dtype=torch.float32,
                device=device,
            )

            model.zero_grad()
            hidden, cell = model.init_hidden()
            model.hidden = hidden.cuda(), hidden.cuda()

            forecasts = model(input_data)

            loss = loss_function(forecasts, target)
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            print('Epoch: {}, loss: {}'.format(epoch, loss.data))

    print('Loss: {}'.format(loss.data))

    return model


def train_fclstm_aqi(grid_seqs, aqi_seqs, hidden_size=1024, learning_rate=0.05):
    model = LSTMAQI(hidden_size, 10).cuda()
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), learning_rate)

    for epoch in range(100):
        for i in range(len(grid_seqs) - 1):
            grid_seq = grid_seqs[i]
            aqi_seq = aqi_seqs[i]

            input_data = torch.tensor(
                grid_seq[:, :-1, :, :, :],  # Remove the last in the seq
                dtype=torch.float32,
                device=device,
            )
            target = torch.tensor(
                aqi_seq[:, 1:, :],
                dtype=torch.float32,
                device=device,
            )

            model.zero_grad()
            hidden, cell = model.init_hidden()
            model.hidden = hidden.cuda(), hidden.cuda()

            forecasts = model(input_data)

            loss = loss_function(forecasts, target)
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            print('Epoch: {}, loss: {}'.format(epoch, loss.data))

    print('Loss: {}'.format(loss.data))

    return model
