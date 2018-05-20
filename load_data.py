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


# output grid meo and aqi data stacked
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


def loadBeijingAqiDataVec():
    dict_data = loadBeijingAqiData()
    ts_data = {}

    for ts in dict_data:
        ts_data[ts] = np.array([])

        for station in sorted(dict_data[ts].keys()):
            ts_data[ts] = np.concatenate((ts_data[ts], dict_data[ts][station]))

    return ts_data
