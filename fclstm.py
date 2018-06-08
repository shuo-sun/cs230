import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import global_consts as cnst


class LSTMForecast(nn.Module):
    def __init__(self, hidden_dim):
        super(LSTMForecast, self).__init__()
        self.hidden_dim = hidden_dim
        self.hidden = self.init_hidden()
        self.conv1_ch = 16
        self.conv2_ch = 32

        self.conv1 = nn.Conv2d(11, self.conv1_ch, 3)  # 29, 19
        self.conv2 = nn.Conv2d(self.conv1_ch, self.conv2_ch, 5, stride=2)  # 13, 8

        self.lstm = nn.LSTM(13 * 8 * 32, hidden_dim, num_layers=3, batch_first=True)

        self.output_meo_size = cnst.BJ_HEIGHT * cnst.BJ_WIDTH * 11
        self.output_aqi_size = cnst.BJ_NUM_AQI_STATIONS * 6
        self.output_meo = nn.Linear(hidden_dim, self.output_meo_size)
        self.output_aqi = nn.Linear(hidden_dim, self.output_aqi_size)

    def forward(self, x, hidden=None):
        m, Tx, n_c, n_h, n_w = x.shape

        if hidden:
            self.hidden = hidden
        else:
            self.hidden = self.init_hidden()

        x = x.view(Tx * m, n_c, n_h, n_w)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x = x.view(m, Tx, 13 * 8 * self.conv2_ch)

        lstm_out, self.hidden = self.lstm(x, self.hidden)
        meo = self.output_meo(lstm_out)
        meo = meo.view(m, Tx, n_c, n_h, n_w)

        aqi = self.output_aqi(lstm_out)
        aqi = aqi.view(m, Tx, 210)

        return torch.cat((meo, aqi), -1)

    def init_hidden(self, batch_size):
        return (torch.zeros(3, batch_size, self.hidden_dim),
                torch.zeros(3, batch_size, self.hidden_dim))
