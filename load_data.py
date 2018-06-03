import numpy as np
import csv
import global_consts as cnst


# output format: a dictionary of ts: meteo grid
def loadBeijingMeoData(rows_to_load=-1):
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
            # TODO: Handle when wind direction is invalid
            time_data[ts][:, y, x] = np.array([float(x) for x in row[4:]])

        return time_data


# output format: a dictionary of dictionary. ts: station: values
def loadBeijingAqiData(rows_to_load=-1):
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

            # TODO: confirm: Put non-provided values as 0
            time_data[ts][station] = np.array([float(x) if len(x) > 0 else 0.0 for x in row[2:]])

        return time_data


# output format: a dictionary from stationId to longitude, latitude
def loadBeijingAqiStationLocations():
    with open('data/beijing_aqi_stations.csv', 'r') as data_file:
        csv_data = csv.reader(data_file, delimiter=',')
        location = {}

        for row in csv_data:
            # Row format: stationId, longitude, latitude

            location[row[0]] = (float(row[1]), float(row[2]))

        return location


# output a dictionary from ts to grid meo and aqi data stacked
def loadBeijingFullData():
    meo_data = loadBeijingMeoData()
    aqi_data = loadBeijingAqiData()
    station_loc = loadBeijingAqiStationLocations()
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
def loadBeijingAqiDataVec():
    dict_data = loadBeijingAqiData()
    stations = sorted(loadBeijingAqiStationLocations().keys())
    ts_data = {}

    for ts in dict_data:
        if ts >= '2017-11-24':
            # The remaining data is invalid
            break

        data = []

        for station in stations:
            if station in dict_data[ts]:
                data.append(dict_data[ts][station])
            else:
                data.append(np.zeros(6))

        ts_data[ts] = np.concatenate(np.array(data), axis=0)

    return ts_data


# Return aligned batched sequences of grids and aqi vectors. With the batch number comes first
def load_batch_seq_data(seq_days=3, batch_size=10):
    time_data = loadBeijingFullData()
    aqi_time_data = loadBeijingAqiDataVec()

    ts_list = sorted(time_data.keys())
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

        if date == '2017-01-02':
            # The first date is incomplete
            continue

        if date >= '2017-11-24':
            # The remaining data is invalid
            break

        if date != prev_date:
            prev_date = date
            day_count += 1

            if day_count >= seq_days:
                if len(sequence) < seq_days * 24:
                    # There are missing ts, which should not happen often.
                    # Padding the remaining ts with 0
                    for i in range(seq_days * 24 - len(sequence)):
                        sequence.append(np.zeros((11, cnst.BJ_HEIGHT, cnst.BJ_WIDTH)))

                if len(aqi_sequence) < seq_days * 24:
                    # There are missing ts, which should not happen often.
                    # Padding the remaining ts with 0
                    for i in range(seq_days * 24 - len(aqi_sequence)):
                        aqi_sequence.append(np.zeros((6 * cnst.BJ_NUM_AQI_STATIONS)))

                sequences.append(sequence)
                sequence = []

                aqi_sequences.append(aqi_sequence)
                aqi_sequence = []

        sequence.append(time_data[ts])
        aqi_sequence.append(aqi_time_data[ts])

    day_count += 1
    if day_count >= seq_days:
        if len(sequence) < seq_days * 24:
            # There are missing ts, which should not happen often.
            # Padding the remaining ts with 0
            for i in range(seq_days * 24 - len(sequence)):
                sequence.append(np.zeros((11, cnst.BJ_HEIGHT, cnst.BJ_WIDTH)))
        sequences.append(sequence)

    if len(aqi_sequence) < seq_days * 24:
        # There are missing ts, which should not happen often.
        # Padding the remaining ts with 0
        for i in range(seq_days * 24 - len(aqi_sequence)):
            aqi_sequence.append(np.zeros((6 * cnst.BJ_NUM_AQI_STATIONS)))

    i = 0
    while i < len(sequences):
        batch = np.array(sequences[i: i + batch_size])
        batches.append(batch)
        aqi_batch = np.array(aqi_sequences[i: i + batch_size])
        aqi_batches.append(aqi_batch)

        i += batch_size

    return batches, aqi_batches
