import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import load_data as ld
import convlstm as md

device = torch.device('cuda')


def compute_dev_set_loss(model, input_seqs, target_meo_seqs, target_aqi_seqs):
    loss_function = nn.MSELoss()
    losses = []

    with torch.no_grad():
        for i in range(len(input_seqs) - 1):
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
                target_aqi_seqs[i][:, :, :204],
                dtype=torch.float32,
                device=device,
            )

            aqi_forecasts, meo_forecasts, _ = model(input_seq)

            loss_aqi = loss_function(aqi_forecasts[:, :, :204], target_aqi)  # Remove the last station which has no data
            loss_meo = loss_function(meo_forecasts, target_meo)

            # Same as training loss. The main training objective is aqi. Therefore, we adjust the
            # ratio to let aqi weights more (5:1).
            loss = loss_aqi + loss_meo / 5
            losses.append(loss.data)

    return np.average(losses)


def compute_overall_smape(model, grid_ts_data, target_aqi_seqs, invalid_rows):
    smape_scores = []
    ts_list = sorted(grid_ts_data.keys())

    new_day_start_pos = []
    prev_date = ''
    for i in range(len(ts_list)):
        ts = ts_list[i]
        date = ts[:10]

        if date != prev_date:
            new_day_start_pos.append(i)
            prev_date = date
    new_day_start_pos.append(len(ts_list))

    with torch.no_grad():
        for i in range(len(new_day_start_pos) - 3):
            ts_input = ts_list[new_day_start_pos[i]: new_day_start_pos[i + 1]]
            ts_target = ts_list[new_day_start_pos[i + 1]: new_day_start_pos[i + 3]]

            forecast_first_date = ts_target[0][:10]

            input_seq = np.array([grid_ts_data[ts] for ts in ts_input])
            input_aqi = np.array([target_aqi_seqs[ts] for ts in ts_input])
            target_seq = {}
            invalid_target_seq = {}

            for ts in ts_target:
                date = ts[:10]
                if date == forecast_first_date:
                    idx = int(ts[11:13])
                else:
                    idx = int(ts[11:13]) + 24

                target_seq[idx] = target_aqi_seqs[ts]
                invalid_target_seq[idx] = invalid_rows[ts]

            # Normal
            normalized_input = normalize_input_seq(input_seq)
            aqi_avg, aqi_std = get_aqi_seq_norm(input_aqi)

            Tx = len(normalized_input)

            normalized_input_tensor = torch.tensor(
                np.reshape(normalized_input[:-1], (1, Tx - 1, 11, 21, 31)),
                dtype=torch.float32,
                device=device,
            )

            normalized_init_tensor = torch.tensor(
                np.reshape(normalized_input[-1], (1, 1, 11, 21, 31)),
                dtype=torch.float32,
                device=device,
            )

            model, hidden_states = feed_model_data(model, normalized_input_tensor)
            aqi_forecasts = generate_forecasts(model, hidden_states, normalized_init_tensor)
            smape_score = \
                compute_smape_score(target_seq, aqi_forecasts, invalid_target_seq, aqi_avg, aqi_std)
            smape_scores.append(smape_score)

    # Take the average of smallest 25 days
    smape_scores.sort()
    return np.mean(smape_scores[:25])


def normalize_input_seq(input_seq):
    Tx, n_c, n_h, n_w = input_seq.shape
    avg = np.mean(input_seq, axis=(0, 2, 3))
    std = np.std(input_seq, axis=(0, 2, 3))

    avg = np.reshape(avg, (1, n_c, 1, 1))
    std = np.reshape(std, (1, n_c, 1, 1))

    return (input_seq - avg) / std


def get_aqi_seq_norm(input_aqi_seq):
    Tx, n_c = input_aqi_seq.shape
    avg = np.mean(input_aqi_seq, axis=0)
    std = np.std(input_aqi_seq, axis=0)

    avg = np.reshape(avg, (1, n_c))
    std = np.reshape(std, (1, n_c))

    return avg, std


def feed_model_data(model, grid_data_seq):
    hidden_states = None

    _, _, hidden_states = model(grid_data_seq, hidden_states)

    return model, hidden_states


# init_grid_data takes the 23:00:00 data
def generate_forecasts(model, hidden_states, init_grid_data, seq_len=48):
    prev_grid = init_grid_data
    aqi_forecasts = {}

    for i in range(seq_len):
        aqi_forecast, prev_grid, hidden_states = model(prev_grid, hidden_states)

        aqi_forecasts[i] = aqi_forecast.cpu().numpy()

    return aqi_forecasts


def compute_smape_score(target_seq, forecast_seq, invalid_rows, avg, std):
    assert len(target_seq) == len(invalid_rows)
    col_mask = [True, True, False, False, True, False]  # Only count PM2.5, PM10, and O3

    scores = []

    for i in target_seq:
        target = target_seq[i]
        forecast = forecast_seq[i] * std + avg
        invalids = invalid_rows[i]

        mask = np.array([True] * 35)

        if len(invalids) == 35:
            continue

        for invalid_row in invalids:
            mask[invalid_row] = False

        target = np.reshape(target, (35, 6))[mask][:, col_mask]
        forecast = np.reshape(forecast, (35, 6))[mask][:, col_mask]

        numerator = np.abs(target - forecast)
        denominator = (target + forecast)

        score = np.mean(numerator / denominator)
        scores.append(score)

    return np.mean(scores)
