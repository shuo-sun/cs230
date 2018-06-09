import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import global_consts as cnst
from load_data import load_bj_aqi_station_locations


class ConvLSTM(nn.Module):
    def __init__(
            self,
            input_size,
            input_channel,
            hidden_channel,
            kernel_size,
            stride=1,
            padding=0):
        """
        Initializations

        :param input_size: (int, int): height, width tuple of the input
        :param input_channel: int: number of channels of the input
        :param hidden_channel: int: number of channels of the hidden state
        :param kernel_size: int: size of the filter
        :param stride: int: stride
        :param padding: int: width of the 0 padding
        """

        super(ConvLSTM, self).__init__()
        self.n_h, self.n_w = input_size
        self.n_c = input_channel
        self.hidden_channel = hidden_channel

        self.conv_xi = nn.Conv2d(
            self.n_c,
            self.hidden_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        self.conv_xf = nn.Conv2d(
            self.n_c,
            self.hidden_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        self.conv_xo = nn.Conv2d(
            self.n_c,
            self.hidden_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        self.conv_xg = nn.Conv2d(
            self.n_c,
            self.hidden_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        self.conv_hi = nn.Conv2d(
            self.hidden_channel,
            self.hidden_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        self.conv_hf = nn.Conv2d(
            self.hidden_channel,
            self.hidden_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        self.conv_ho = nn.Conv2d(
            self.hidden_channel,
            self.hidden_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        self.conv_hg = nn.Conv2d(
            self.hidden_channel,
            self.hidden_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        self.conv_hi = nn.Conv2d(
            self.hidden_channel,
            self.hidden_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

    def forward(self, x, hidden_states):
        """
        Forward prop.

        :param x: input tensor of shape (n_batch, n_c, n_h, n_w)
        :param hidden_states: (tensor, tensor) for hidden and cell states.
                              Each of shape (n_batch, n_hc, n_hh, n_hw)
        :return: (hidden_state, cell_state)
        """

        hidden_state, cell_state = hidden_states

        xi = self.conv_xi(x)
        hi = self.conv_hi(hidden_state)
        xf = self.conv_xf(x)
        hf = self.conv_hf(hidden_state)
        xo = self.conv_xo(x)
        ho = self.conv_ho(hidden_state)
        xg = self.conv_xg(x)
        hg = self.conv_hg(hidden_state)

        i = torch.sigmoid(xi + hi)
        f = torch.sigmoid(xf + hf)
        o = torch.sigmoid(xo + ho)
        g = torch.tanh(xg + hg)

        cell_state = f * cell_state + i * g
        hidden_state = o * torch.tanh(cell_state)

        return hidden_state, cell_state

    def init_hidden(self, batch_size):
        return (torch.zeros(batch_size, self.hidden_channel, self.n_h, self.n_w).cuda(),
                torch.zeros(batch_size, self.hidden_channel, self.n_h, self.n_w).cuda())


class ConvLSTMForecast(nn.Module):
    def __init__(self, input_size, hidden_dim, kernel_size, padding):
        """
        Init function.

        :param input_size: (int, int): input h, w
        """

        super(ConvLSTMForecast, self).__init__()

        self.pred_box_size = 1  # prediction looks at 2 * 1 + 1 box
        self.pred_box_w = 2 * self.pred_box_size + 1

        self.convlstm1 = ConvLSTM(
            input_size,
            11,
            hidden_dim,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
        )

        self.convlstm2 = ConvLSTM(
            input_size,
            hidden_dim,
            hidden_dim,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
        )

        self.convlstm3 = ConvLSTM(
            input_size,
            hidden_dim,
            hidden_dim,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
        )

        (self.hidden1, self.cell1), (self.hidden2, self.cell2), (self.hidden3, self.cell3) = \
            self.init_hidden(5)

        self.meo_conv_1 = nn.Conv2d(hidden_dim, hidden_dim, 1)
        self.meo_conv_output = nn.Conv2d(hidden_dim, 11, 1)  # The final meo prediction layer

        self.aqi_fc_1 = nn.Linear(hidden_dim * (self.pred_box_w ** 2), 256)
        self.aqi_fc_output = nn.Linear(256, 6)

        # find close cells related to AQI stations
        station_loc = load_bj_aqi_station_locations()
        self.station_order = sorted(station_loc.keys())
        self.station_locs = []  # top left y, x position

        for station in self.station_order:
            long, lat = station_loc[station]
            y_c, x_c = (
                int(10 * lat + 0.5 - 10 * cnst.BJ_LATITUDE_START),
                int(10 * long + 0.5 - 10 * cnst.BJ_LONGITUDE_START)
            )

            # Handle when the station is close to boundaries
            y = min(max(y_c - self.pred_box_size, 0), cnst.BJ_HEIGHT - self.pred_box_w)
            x = min(max(x_c - self.pred_box_size, 0), cnst.BJ_WIDTH - self.pred_box_w)

            self.station_locs.append((y, x))

    def forward(self, X, hidden_states=None):
        m, Tx, n_c, n_h, n_w = X.shape

        meo_output = []
        aqi_output = []

        if hidden_states:
            (self.hidden1, self.cell1), (self.hidden2, self.cell2), (self.hidden3, self.cell3) = \
                hidden_states
        else:
            (self.hidden1, self.cell1), (self.hidden2, self.cell2), (self.hidden3, self.cell3) = \
                self.init_hidden(m)

        for t in range(Tx):
            xt = X[:, t, :, :, :]
            self.hidden1, self.cell1 = self.convlstm1(xt, (self.hidden1, self.cell1))
            self.hidden2, self.cell2 = self.convlstm2(self.hidden1, (self.hidden2, self.cell2))
            self.hidden3, self.cell3 = self.convlstm3(self.hidden2, (self.hidden3, self.cell3))

            # MEO prediction
            meo_pred = torch.tanh(self.meo_conv_1(self.hidden3))
            meo_pred = self.meo_conv_output(meo_pred)
            meo_pred = meo_pred.view(m, 1, 11, n_h, n_w)
            meo_output.append(meo_pred)

            cell_list = []
            for (y, x) in self.station_locs:
                cells = \
                    self.hidden3[:, :, y:y+self.pred_box_w, x:x+self.pred_box_w]\
                        .contiguous().view(m, 1, -1)
                cell_list.append(cells)
                
            num_stations = len(cell_list)
            cells = torch.cat(cell_list, dim=1).view(m * num_stations, -1)
            cells = torch.tanh(self.aqi_fc_1(cells))
            aqi = self.aqi_fc_output(cells)
            aqi_stations = aqi.view(m, 1, num_stations * 6)

            aqi_output.append(aqi_stations)

        # cat on time dimension
        meo_output = torch.cat(meo_output, dim=1)
        aqi_output = torch.cat(aqi_output, dim=1)

        hidden_states = (self.hidden1, self.cell1), (self.hidden2, self.cell2), (self.hidden3, self.cell3)

        return aqi_output, meo_output, hidden_states

    def init_hidden(self, batch_size):
        return self.convlstm1.init_hidden(batch_size), \
               self.convlstm2.init_hidden(batch_size), \
               self.convlstm3.init_hidden(batch_size)


class ConvLSTMForecast2L(nn.Module):
    def __init__(self, input_size, hidden_dim, kernel_size, padding):
        """
        Init function.

        :param input_size: (int, int): input h, w
        """

        super(ConvLSTMForecast2L, self).__init__()

        self.pred_box_size = 1  # prediction looks at 2 * 1 + 1 box
        self.pred_box_w = 2 * self.pred_box_size + 1

        self.convlstm1 = ConvLSTM(
            input_size,
            11,
            hidden_dim,  # 128
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
        )

        self.convlstm2 = ConvLSTM(
            input_size,
            hidden_dim,
            hidden_dim,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
        )

        (self.hidden1, self.cell1), (self.hidden2, self.cell2) = self.init_hidden(10)

        self.meo_conv_1 = nn.Conv2d(hidden_dim, hidden_dim, 1)
        self.meo_conv_output = nn.Conv2d(hidden_dim, 11, 1)  # The final meo prediction layer

        self.aqi_fc_1 = nn.Linear(hidden_dim * (self.pred_box_w ** 2), hidden_dim)
        self.aqi_fc_output = nn.Linear(hidden_dim, 6)

        # find close cells related to AQI stations
        station_loc = load_bj_aqi_station_locations()
        self.station_order = sorted(station_loc.keys())
        self.station_locs = []  # top left y, x position

        for station in self.station_order:
            long, lat = station_loc[station]
            y_c, x_c = (
                int(10 * lat + 0.5 - 10 * cnst.BJ_LATITUDE_START),
                int(10 * long + 0.5 - 10 * cnst.BJ_LONGITUDE_START)
            )

            # Handle when the station is close to boundaries
            y = min(max(y_c - self.pred_box_size, 0), cnst.BJ_HEIGHT - self.pred_box_w)
            x = min(max(x_c - self.pred_box_size, 0), cnst.BJ_WIDTH - self.pred_box_w)

            self.station_locs.append((y, x))

    def forward(self, X, hidden_states=None):
        m, Tx, n_c, n_h, n_w = X.shape

        meo_output = []
        aqi_output = []

        if hidden_states:
            (self.hidden1, self.cell1), (self.hidden2, self.cell2) = hidden_states
        else:
            (self.hidden1, self.cell1), (self.hidden2, self.cell2) = self.init_hidden(m)

        for t in range(Tx):
            xt = X[:, t, :, :, :]
            self.hidden1, self.cell1 = self.convlstm1(xt, (self.hidden1, self.cell1))
            self.hidden2, self.cell2 = self.convlstm2(self.hidden1, (self.hidden2, self.cell2))

            # MEO prediction
            meo_pred = torch.tanh(self.meo_conv_1(self.hidden2))
            meo_pred = self.meo_conv_output(meo_pred)
            meo_pred = meo_pred.view(m, 1, 11, n_h, n_w)
            meo_output.append(meo_pred)

            cell_list = []
            for (y, x) in self.station_locs:
                cells = \
                    self.hidden2[:, :, y:y + self.pred_box_w, x:x + self.pred_box_w] \
                        .contiguous().view(m, 1, -1)
                cell_list.append(cells)

            num_stations = len(cell_list)
            cells = torch.cat(cell_list, dim=1).view(m * num_stations, -1)
            cells = torch.tanh(self.aqi_fc_1(cells))
            aqi = self.aqi_fc_output(cells)
            aqi_stations = aqi.view(m, 1, num_stations * 6)

            aqi_output.append(aqi_stations)

        # cat on time dimension
        meo_output = torch.cat(meo_output, dim=1)
        aqi_output = torch.cat(aqi_output, dim=1)

        hidden_states = (self.hidden1, self.cell1), (self.hidden2, self.cell2)

        return aqi_output, meo_output, hidden_states

    def init_hidden(self, batch_size):
        return self.convlstm1.init_hidden(batch_size), \
               self.convlstm2.init_hidden(batch_size)
