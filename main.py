import numpy as np
import load_data as ld
import convlstm as md
import torch
import convlstm_training as tr
import evaluation as ev


def main():
    # Training data
    grid_seqs, aqi_seqs = ld.load_batch_seq_data()
    input_seqs, target_meo_seqs, target_aqi_seqs = tr.seq_preprocessing(grid_seqs, aqi_seqs)

    # Dev data
    dev_grid_seqs, dev_aqi_seqs = ld.load_batch_dev_seq_data()
    dev_input_seqs, dev_target_meo_seqs, dev_target_aqi_seqs = \
        tr.seq_preprocessing(dev_grid_seqs, dev_aqi_seqs)

    # Dev data for SMAPE calculation
    dev_grid_ts_data, _ = ld.load_dev_full_data()
    dev_target_aqi_ts, invalid_rows = ld.load_dev_aqi_data_vec()

    model = md.ConvLSTMForecast2L((21, 31), 256, 3, 1).cuda()
    snapshots = []

    model = tr.train(
        model, input_seqs, target_meo_seqs, target_aqi_seqs, dev_input_seqs, dev_target_meo_seqs, dev_target_aqi_seqs,
        dev_grid_ts_data, dev_target_aqi_ts, invalid_rows, snapshots, iterations=5, lr=0.01)

    model = tr.train(
        model, input_seqs, target_meo_seqs, target_aqi_seqs, dev_input_seqs, dev_target_meo_seqs, dev_target_aqi_seqs,
        dev_grid_ts_data, dev_target_aqi_ts, invalid_rows, snapshots, iterations=10, lr=0.001)

    # Test data
    test_grid_seqs, test_aqi_seqs = ld.load_batch_test_seq_data()
    test_input_seqs, test_target_meo_seqs, test_target_aqi_seqs = \
        tr.seq_preprocessing(test_grid_seqs, test_aqi_seqs)

    # Dev data for SMAPE calculation
    test_grid_ts_data, _ = ld.load_test_full_data()
    test_target_aqi_ts, invalid_rows = ld.load_test_aqi_data_vec()

    test_loss = ev.compute_dev_set_loss(
        model,
        test_input_seqs,
        test_target_meo_seqs,
        test_target_aqi_seqs)

    test_smape = ev.compute_overall_smape(
        model,
        test_grid_ts_data,
        test_target_aqi_ts,
        invalid_rows)

    print('Test loss: {}, test SMAPE: {}'.format(test_loss, test_smape))


if __name__ == "__main__":
    main()
