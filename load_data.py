import numpy as np
import csv
import global_consts as cnst

def loadBeijingMeoData():
    with open('data/Beijing_historical_meo_grid.csv', 'r') as data_file:
        next(data_file)
        csv_data = csv.reader(data_file, delimiter=',')
        time_data = {}
        prev_ts = ''

        for row in csv_data:
            # Row format: grid name, longitude, latitude, time,
            # temperature, pressure, humidity, wind_direction, wind_speed/kph

            ts = row[3]
            if ts != prev_ts:
                prev_ts = ts
                time_data[ts] = np.zeros((cnst.BJ_HEIGHT, cnst.BJ_WIDTH, 5))

            y = int((10 * float(row[2]) - 10 * cnst.BJ_LATITUDE_START))
            x = int((10 * float(row[1]) - 10 * cnst.BJ_LONGITUDE_START))
            # TODO: Handle when wind direction is invalid
            time_data[ts][y, x] = np.array([float(x) for x in row[4:]])

        return time_data


def loadBeijingAqiData():
    with open('data/beijing_17_18_aq.csv', 'r') as data_file:
        next(data_file)
        csv_data = csv.reader(data_file, delimiter=',')
        time_data = {}

        for row in csv_data:
            # Row format: stationId, time, pm2.5, pm10, no2, co, o3, so2

            station = row[0]
            ts = row[1]

            if ts not in time_data:
                time_data[ts] = {}

            # Put non-provided values as -1
            time_data[ts][station] = np.array([float(x) if len(x) > 0 else -1 for x in row[2:]])
