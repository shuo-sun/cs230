import numpy as np
import csv
import global_consts as cnst


# output format: a dictionary of ts: meteo grid
def load_bj_meo_data(rows_to_load=-1):
    with open('data/Beijing_historical_meo_grid.csv', 'r') as data_file:
        next(data_file)
        csv_data = csv.reader(data_file, delimiter=',')
        time_data = {}
        prev_ts = ''

        count = 0

        for row in csv_data:
            # Row format: grid name, longitude, latitude, time,
            # temperature, pressure, humidity, wind_direction, wind_speed/kph

            if rows_to_load != -1 and count >= rows_to_load:
                break
            count += 1

            ts = row[3]
            if ts != prev_ts:
                prev_ts = ts
                time_data[ts] = np.zeros((5, cnst.BJ_HEIGHT, cnst.BJ_WIDTH))

            y = int((10 * float(row[2]) - 10 * cnst.BJ_LATITUDE_START))
            x = int((10 * float(row[1]) - 10 * cnst.BJ_LONGITUDE_START))

            time_data[ts][:, y, x] = np.array([float(x) for x in row[4:]])

        return time_data


# output format: a dictionary of dictionary. ts: station: values
def load_bj_aqi_data(rows_to_load=-1):
    with open('data/beijing_17_18_aq.csv', 'r') as data_file:
        next(data_file)
        csv_data = csv.reader(data_file, delimiter=',')
        time_data = {}

        count = 0

        for row in csv_data:
            # Row format: stationId, time, pm2.5, pm10, no2, co, o3, so2

            if rows_to_load != -1 and count >= rows_to_load:
                break
            count += 1

            station = row[0]
            ts = row[1]

            if ts not in time_data:
                time_data[ts] = {}

            data = []
            for i in range(2, len(row)):
                x = row[i]
                if len(x) > 0:
                    data.append(float(x))
                elif ts > '2017-01-01 14:00:00':  # Not the first day
                    # if no data provided, use the previous data
                    data.append(time_data[prev_ts][station][i - 2])
                else:
                    data.append(0.01)

            time_data[ts][station] = np.array(data)

            prev_ts = ts

        return time_data


# output format: a dictionary from stationId to longitude, latitude
def load_bj_aqi_station_locations():
    with open('data/beijing_aqi_stations.csv', 'r') as data_file:
        csv_data = csv.reader(data_file, delimiter=',')
        location = {}

        for row in csv_data:
            # Row format: stationId, longitude, latitude

            location[row[0]] = (float(row[1]), float(row[2]))

        return location


# output a dictionary from ts to grid meo and aqi data stacked
def load_bj_full_data():
    meo_data = load_bj_meo_data()
    aqi_data = load_bj_aqi_data()

    return combine_meo_aqi_data(meo_data, aqi_data)


def combine_meo_aqi_data(meo_data, aqi_data, exclude_stations={}):
    station_loc = load_bj_aqi_station_locations()
    time_data = {}

    print("Finished loading raw data...")

    aqi_grid_long = np.zeros((1, cnst.BJ_HEIGHT, cnst.BJ_WIDTH))
    aqi_grid_lat = np.zeros((1, cnst.BJ_HEIGHT, cnst.BJ_WIDTH))

    for i in range(cnst.BJ_HEIGHT):
        for j in range(cnst.BJ_WIDTH):
            aqi_grid_long[0, i, j] = cnst.BJ_LONGITUDE_START + float(i) / 10
            aqi_grid_lat[0, i, j] = cnst.BJ_LATITUDE_START + float(j) / 10

    for ts in aqi_data.keys():
        if ts not in meo_data:  # ts is not strictly aligned
            continue

        aqi_grid = np.zeros((6, cnst.BJ_HEIGHT, cnst.BJ_WIDTH))
        meo_grid = meo_data[ts]
        aqi = aqi_data[ts]
        sum_weights = np.zeros((1, cnst.BJ_HEIGHT, cnst.BJ_WIDTH))

        for station, value in aqi.items():
            if station in exclude_stations:
                continue

            long_station, lat_station = station_loc[station]

            long_diff = np.abs(aqi_grid_long - long_station)
            lat_diff = np.abs(aqi_grid_lat - lat_station)
            dist_squared = long_diff ** 2 + lat_diff ** 2 + 10 ** (-8)  # prevent divide by zero

            weights = 1 / dist_squared

            aqi_grid = aqi_grid + weights * np.reshape(value, (6, 1, 1))
            sum_weights = sum_weights + weights

        aqi_grid = aqi_grid / sum_weights
        time_data[ts] = np.concatenate((meo_grid, aqi_grid), axis=0)

    return time_data


# Return a dict from ts to a matrix of station data. Stations are in alphabetic order
def load_bj_aqi_data_vec():
    dict_data = load_bj_aqi_data()
    return convert_aqi_to_vec(dict_data)[0]


def convert_aqi_to_vec(dict_data, invalid_rows=[]):
    stations = sorted(load_bj_aqi_station_locations().keys())
    ts_data = {}
    invalid_rows = {x: 0 for x in invalid_rows}
    ts_invalid_row_nums = {}

    for ts in dict_data:
        data = []
        invalid_row_nums = []

        for station in stations:
            if station in dict_data[ts]:
                if (ts, station) in invalid_rows:
                    invalid_row_nums.append(len(data))

                data.append(dict_data[ts][station])
            else:
                invalid_row_nums.append(len(data))

                data.append(np.zeros(6))

        ts_data[ts] = np.concatenate(np.array(data), axis=0)
        ts_invalid_row_nums[ts] = invalid_row_nums

    return ts_data, ts_invalid_row_nums


# Return aligned batched sequences of grids and aqi vectors. With the batch number comes first
def load_batch_seq_data(seq_days=3, batch_size=5):
    grid_time_data = load_bj_full_data()
    aqi_time_data = load_bj_aqi_data_vec()

    return make_batch_seq_data(grid_time_data, aqi_time_data, seq_days, batch_size)


def make_batch_seq_data(grid_time_data, aqi_time_data, seq_days, batch_size):
    ts_list = sorted(grid_time_data.keys())
    sequences = []
    aqi_sequences = []
    sequence = []
    aqi_sequence = []
    batches = []
    aqi_batches = []  # Same batch as input batches. But for data of AQI stations

    day_count = 0
    prev_date = '2017-01-02'  # The first date is incomplete

    for ts in ts_list:
        date = ts[:10]

        if date == '2017-01-01':
            # The first date is incomplete
            continue

        if date != prev_date:
            prev_date = date
            day_count += 1

            if day_count >= seq_days:
                if len(sequence) < seq_days * 24:
                    # There are missing ts, which should not happen often.
                    # Padding the remaining ts with previous values
                    for i in range(seq_days * 24 - len(sequence)):
                        sequence.append(np.copy(sequence[-1]))

                if len(aqi_sequence) < seq_days * 24:
                    # There are missing ts, which should not happen often.
                    # Padding the remaining ts with previous values
                    for i in range(seq_days * 24 - len(aqi_sequence)):
                        aqi_sequence.append(np.copy(aqi_sequence[-1]))

                sequences.append(sequence)
                sequence = []

                aqi_sequences.append(aqi_sequence)
                aqi_sequence = []

        sequence.append(grid_time_data[ts])
        aqi_sequence.append(aqi_time_data[ts])

    day_count += 1
    if day_count >= seq_days:
        if len(sequence) < seq_days * 24:
            # There are missing ts, which should not happen often.
            # Padding the remaining ts with 0
            for i in range(seq_days * 24 - len(sequence)):
                sequence.append(np.copy(sequence[-1]))
        sequences.append(sequence)

        if len(aqi_sequence) < seq_days * 24:
            # There are missing ts, which should not happen often.
            # Padding the remaining ts with 0
            for i in range(seq_days * 24 - len(aqi_sequence)):
                aqi_sequence.append(np.copy(aqi_sequence[-1]))
        aqi_sequences.append(aqi_sequence)

    i = 0
    while i < len(sequences):
        batch = np.array(sequences[i: i + batch_size])
        batches.append(batch)
        aqi_batch = np.array(aqi_sequences[i: i + batch_size])
        aqi_batches.append(aqi_batch)

        i += batch_size

    return batches, aqi_batches


def load_bj_meo_dev_data():
    with open('data/beijing_meo_04-05.csv', 'r') as data_file:
        next(data_file)
        csv_data = csv.reader(data_file, delimiter=',')
        time_data = {}
        prev_ts = ''

        for row in csv_data:
            # Row format: station_id, grid name, time, weather,
            # temperature, pressure, humidity, wind_direction, wind_speed/kph

            ts = row[2]

            if ts >= '2018-05-01':
                break

            if ts != prev_ts:
                prev_ts = ts
                time_data[ts] = np.zeros((5, cnst.BJ_HEIGHT, cnst.BJ_WIDTH))

            long, lat = cnst.grid_to_loc[row[1]]

            y = int((10 * lat - 10 * cnst.BJ_LATITUDE_START))
            x = int((10 * long - 10 * cnst.BJ_LONGITUDE_START))

            time_data[ts][:, y, x] = np.array([float(x) for x in row[4:]])

        return time_data


def load_bj_aqi_dev_data():
    invalid_rows = []

    with open('data/beijing_aqi_04-05.csv', 'r') as data_file:
        next(data_file)
        csv_data = csv.reader(data_file, delimiter=',')
        time_data = {}

        prev_ts = None
        last_ts = None

        for row in csv_data:
            # Row format: id, stationId, time, pm2.5, pm10, no2, co, o3, so2

            station = row[1]
            ts = row[2]

            if ts >= '2018-05-01':
                break

            if ts not in time_data:
                time_data[ts] = {}
                prev_ts = last_ts

            data = []
            for i in range(3, len(row)):
                x = row[i]
                if len(x) > 0:
                    data.append(float(x))
                else:
                    if ts > '2018-04-01 01:00:00':  # Not the first ts
                        # if no data provided, use the previous data
                        data.append(time_data[prev_ts][station][i - 3])
                    else:
                        data.append(0.01)

                    invalid_rows.append((ts, station))

            time_data[ts][station] = np.array(data)

            last_ts = ts

        return time_data, invalid_rows


def load_bj_meo_test_data():
    with open('data/beijing_meo_04-05.csv', 'r') as data_file:
        next(data_file)
        csv_data = csv.reader(data_file, delimiter=',')
        time_data = {}
        prev_ts = ''

        for row in csv_data:
            # Row format: station_id, grid name, time, weather
            # temperature, pressure, humidity, wind_direction, wind_speed/kph

            ts = row[2]

            if ts < '2018-05-01':
                continue

            if ts != prev_ts:
                prev_ts = ts
                time_data[ts] = np.zeros((5, cnst.BJ_HEIGHT, cnst.BJ_WIDTH))

            long, lat = cnst.grid_to_loc[row[1]]

            y = int((10 * lat - 10 * cnst.BJ_LATITUDE_START))
            x = int((10 * long - 10 * cnst.BJ_LONGITUDE_START))

            time_data[ts][:, y, x] = np.array([float(x) for x in row[4:]])

        return time_data


def load_bj_aqi_test_data():
    invalid_rows = []

    with open('data/beijing_aqi_04-05.csv', 'r') as data_file:
        next(data_file)
        csv_data = csv.reader(data_file, delimiter=',')
        time_data = {}

        for row in csv_data:
            # Row format: id, stationId, time, pm2.5, pm10, no2, co, o3, so2

            station = row[1]
            ts = row[2]

            if ts < '2018-05-01':
                continue

            if ts not in time_data:
                time_data[ts] = {}

            data = []
            for i in range(3, len(row)):
                x = row[i]
                if len(x) > 0:
                    data.append(float(x))
                else:
                    data.append(-0.01)
                    invalid_rows.append((ts, station))

            time_data[ts][station] = np.array(data)

        for station in sorted(load_bj_aqi_station_locations().keys()):
            prev_ts = None
            for ts in sorted(time_data.keys()):
                data = time_data[ts][station]
                for i in range(len(data)):
                    if data[i] < 0 and prev_ts is not None:
                        data[i] = time_data[prev_ts][station][i]

        return time_data, invalid_rows


def load_dev_full_data():
    meo_data = load_bj_meo_dev_data()
    aqi_data, invalid_rows = load_bj_aqi_dev_data()

    # Stations with no valid data
    exclude_stations = {
        'zhiwuyuan_aq': 0
    }

    return combine_meo_aqi_data(meo_data, aqi_data, exclude_stations), invalid_rows


def load_test_full_data():
    meo_data = load_bj_meo_test_data()
    aqi_data, invalid_rows = load_bj_aqi_test_data()

    exclude_stations = {
        'zhiwuyuan_aq': 0
    }

    return combine_meo_aqi_data(meo_data, aqi_data, exclude_stations), invalid_rows


def load_dev_aqi_data_vec():
    aqi_data, invalid_rows = load_bj_aqi_dev_data()
    return convert_aqi_to_vec(aqi_data, invalid_rows)


def load_test_aqi_data_vec():
    aqi_data, invalid_rows = load_bj_aqi_test_data()
    return convert_aqi_to_vec(aqi_data, invalid_rows)


def load_batch_dev_seq_data(seq_days=3, batch_size=5):
    grid_time_data, _ = load_dev_full_data()
    aqi_time_data, _ = load_dev_aqi_data_vec()

    return make_batch_seq_data(grid_time_data, aqi_time_data, seq_days, batch_size)


def load_batch_test_seq_data(seq_days=3, batch_size=5):
    grid_time_data, _ = load_test_full_data()
    aqi_time_data, _ = load_test_aqi_data_vec()

    return make_batch_seq_data(grid_time_data, aqi_time_data, seq_days, batch_size)
