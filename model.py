import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from load_data import loadBeijingFullData, loadBeijingAqiDataVec
import global_consts as cnst


class LSTMForecast(nn.Module):
    def __init__(self, hidden_dim, batch_size):
        super(LSTMForecast, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.hidden = self.init_hidden()

        self.conv1 = nn.Conv2d(11, 16, 3)  # 29, 19
        self.conv2 = nn.Conv2d(16, 32, 5, stride=2)  # 13, 8

        self.lstm = nn.LSTM(13 * 8 * 32, hidden_dim, 3)

        self.output_meo_size = cnst.BJ_HEIGHT * cnst.BJ_WIDTH * 5
        self.output_aqi_size = 35 * 6
        self.output_meo = nn.Linear(128, self.output_meo_size)
        self.output_aqi = nn.Linear(128, self.output_aqi_size)

    def forward(self, x):
        m = x.shape[0]

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(1, m, 13 * 8 * 32)
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        meo = self.output_meo(lstm_out)
        aqi = self.output_aqi(lstm_out)

        return torch.cat((meo, aqi), -1)

    def init_hidden(self):
        return torch.zeros(3, self.batch_size, self.hidden_dim), torch.zeros(3, self.batch_size, self.hidden_dim)
