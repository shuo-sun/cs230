import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import load_data as ld
import convlstm as md
import evaluation as ev

device = torch.device('cuda')


def seq_preprocessing(grid_seqs, aqi_seqs):
    """

    :param grid_seqs: list of (m, Tx, n_c, n_h, n_w)
    :param aqi_seqs: list of (m, Tx, n_c)
    :return:
    """

    input_seqs = []
    target_meo_seqs = []
    target_aqi_seqs = []

    avg_grids = []
    std_grids = []
    avg_aqis = []
    std_aqis = []

    for data in grid_seqs:
        m, Tx, _, _, _ = data.shape
        avg = np.reshape(np.average(data, axis=(1, 3, 4)), (m, 1, 11, 1, 1))
        std = np.reshape(np.std(data, axis=(1, 3, 4)), (m, 1, 11, 1, 1))
        avg_grids.append(avg)
        std_grids.append(std)

    for data in aqi_seqs:
        m, Tx, _ = data.shape
        avg = np.reshape(np.average(data, axis=(1)), (m, 1, 210))
        std = np.reshape(np.std(data, axis=(1)), (m, 1, 210))
        avg_aqis.append(avg)
        std_aqis.append(std)

    for i in range(len(grid_seqs)):
        grid_seq = grid_seqs[i]
        aqi_seq = aqi_seqs[i]

        grid_seq = (grid_seq - avg_grids[i]) / std_grids[i]
        # Handle 0 std when no data exists
        aqi_seq = aqi_seq - avg_aqis[i]
        aqi_seq = np.divide(aqi_seq, std_aqis[i], out=aqi_seq, where=std_aqis[i] != 0)

        input_seq = grid_seq[:, :-1, :, :, :]  # Remove the last from the input seq

        target_meo = grid_seq[:, 1:, :, :, :]

        target_aqi = aqi_seq[:, 1:, :]

        input_seqs.append(input_seq)
        target_meo_seqs.append(target_meo)
        target_aqi_seqs.append(target_aqi)

    assert len(input_seqs) == len(target_meo_seqs) and len(target_meo_seqs) == len(target_aqi_seqs)

    return input_seqs, target_meo_seqs, target_aqi_seqs


def train(
        model,
        input_seqs,
        target_meo_seqs,
        target_aqi_seqs,
        dev_input_seqs,
        dev_target_meo_seqs,
        dev_target_aqi_seqs,
        smape_grid_ts_data,
        smape_target_aqi_seqs,
        smape_invalid_rows,
        snapshots,
        iterations=100,
        lr=0.01,
        clipping_norm=1e-5):
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr)

    for epoch in range(iterations):
        losses = []
        for i in range(len(input_seqs) - 1):  # The last one is incomplete
            input_seq = torch.tensor(
                input_seqs[i],
                dtype=torch.float32,
                device=device,
            )

            target_meo = torch.tensor(
                target_meo_seqs[i],
                dtype=torch.float32,
                device=device,
            )

            target_aqi = torch.tensor(
                target_aqi_seqs[i],
                dtype=torch.float32,
                device=device,
            )

            model.zero_grad()

            aqi_forecasts, meo_forecasts, _ = model(input_seq)

            loss_aqi = loss_function(aqi_forecasts, target_aqi)
            loss_meo = loss_function(meo_forecasts, target_meo)

            # The main training objective is aqi. Therefore, we adjust the ratio to let aqi
            # weights more (5:1).
            loss = loss_aqi + loss_meo / 5
            loss.backward()

            losses.append(loss.data)

            nn.utils.clip_grad_norm_(model.parameters(), clipping_norm)

            optimizer.step()

        loss = np.average(losses)
        dev_loss = \
            ev.compute_dev_set_loss(model, dev_input_seqs, dev_target_meo_seqs, dev_target_aqi_seqs)
        dev_smape = ev.compute_overall_smape(model, smape_grid_ts_data, smape_target_aqi_seqs, smape_invalid_rows)
        print('Epoch: {}, average loss: {}, dev loss: {}, dev smape: {}'.format(epoch, loss, dev_loss, dev_smape))

        snapshots.append((model.state_dict(), loss, dev_loss, dev_smape))

    print('Loss: {}'.format(loss))

    return model
