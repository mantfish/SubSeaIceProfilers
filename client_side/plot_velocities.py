import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt

if __name__ ==  "__main__":
    file_name = input("input file name: ")
    file_name = "data/" + file_name
    df = pd.read_csv(file_name, names = ["time", "x", "y", "theta"])

    time_series = []
    for time in df["time"]:
        time_series.append(dt.datetime.strptime(time,'%M:%S.%f'))

    plt.plot(time_series,df["theta"])
    plt.show()
    velocity = []

    for index, row in df.iterrows():
        if index > 0:
            delta_pos = np.sqrt((row["x"]-prev_x)**2 + (row["y"]-prev_y)**2)
            print(row["time"])
            delta_time = (dt.datetime.strptime(row["time"],'%M:%S.%f')-dt.datetime.strptime(prev_time,'%M:%S.%f')).microseconds/(1e6)
            velocity.append(delta_pos/delta_time)
        else:
            velocity.append(0)

        prev_x = row["x"]
        prev_y = row["y"]
        prev_time = row["time"]

    plt.plot(time_series,velocity)
    plt.show()

