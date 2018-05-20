import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from load_data import loadBeijingFullData, loadBeijingAqiDataVec
import global_consts as cnst
from model import LSTMForecast


def get_seqs():
    ts_data = loadBeijingFullData()
    aqi_ts_data = loadBeijingAqiDataVec()
    input_seq = []
    output_seq = []

    for ts in sorted(ts_data.keys()):
        input_seq.append(ts_data[ts])
        output_seq.append(np.concatenate((np.reshape(ts_data[ts][:5, :, :], (-1)), aqi_ts_data[ts])))

    input_seq = input_seq[:-1]
    output_seq = output_seq[1:]

    return input_seq, output_seq


def train():
    input_seq, output_seq = get_seqs()

    model = LSTMForecast(128, 1)
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), 0.05)

    for epoch in range(10):
        for i in range(len(input_seq)):
            input_data = torch.tensor(np.reshape(input_seq[i], (1, 11, 31, 21)), dtype=torch.float32)
            target = torch.tensor(np.reshape(output_seq[i], (1, 1, -1)), dtype=torch.float32)

            model.zero_grad()
            model.hidden = model.init_hidden()

            forecasts = model(input_data)

            loss = loss_function(forecasts, target)
            loss.backward()
            optimizer.step()

        if epoch % 100 == 0:
            print('Epoch: {}, loss: {}'.format(epoch, loss.data))

    with torch.no_grad():
        forecasts = model(torch.tensor(np.reshape(input_seq[0], (1, 11, 31, 21)), dtype=torch.float32))
        print(forecasts)
        print(output_seq[0])

train()