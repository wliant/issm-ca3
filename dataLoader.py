import os
import datetime
import random
import pandas as pd
import numpy as np
from geopy import distance

THRESHOLD = 30

def calculate_speed(current, next):
    if next is None:
        return 0
    d = distance.distance((current["latitude"],current["longitude"]),(next["latitude"],next["longitude"])).km * 1000
    time = (next["day_since"]-current["day_since"]) * 24 * 3600
    if time <= 0:
        time = 1
    if time > THRESHOLD:
        return 0
    return d /time
def calculate_acceleration(current, next):
    if next is None:
        return 0
    speed_diff = next["speed"] - current["speed"]
    time = (next["day_since"]-current["day_since"]) * 24 * 3600
    if time <= 0:
        time = 1
    if time > THRESHOLD:
        return 0
    return speed_diff / time

def calculate_jerk(current, next):
    if next is None:
        return 0
    acc_diff = next["acceleration"] - current["acceleration"]
    time = (next["day_since"]-current["day_since"]) * 24 * 3600
    if time <= 0:
        time = 1
    if time > THRESHOLD:
        return 0
    return acc_diff / time

def calculate_bearing(current, next):
    if next is None:
        return 0
    time = (next["day_since"]-current["day_since"]) * 24 * 3600
    if time <= 0:
        time = 1
    if time > THRESHOLD:
        return 0
    X = np.cos(next["latitude"]) * np.sin(next["longitude"]-current["longitude"])
    Y = np.cos(current["latitude"]) * np.sin(next["latitude"]) - np.sin(current["latitude"]) * np.cos(next["latitude"]) * np.cos(next["longitude"] - current["longitude"])
    return np.arctan2(X,Y) * 180/np.pi

def calculate_bearing_rate(current, next):
    if next is None:
        return 0
    time = (next["day_since"]-current["day_since"]) * 24 * 3600
    if time <= 0:
        time = 1
    if time > THRESHOLD:
        return 0
    diff = next["bearing"] - current["bearing"]
    return diff / time

def preprocess(data_points):
    data_points.sort(key=lambda x: x["timestamp"])
    for i in range(0, len(data_points)):
        j = i+1
        next = data_points[j] if j < len(data_points) else None
        data_points[i]["speed"] = calculate_speed(data_points[i], next)

    for i in range(0, len(data_points)):
        j = i+1
        next = data_points[j] if j < len(data_points) else None
        data_points[i]["acceleration"] = calculate_acceleration(data_points[i], next)
    for i in range(0, len(data_points)):
        j = i+1
        next = data_points[j] if j < len(data_points) else None
        data_points[i]["jerk"] = calculate_jerk(data_points[i], next)
    for i in range(0, len(data_points)):
        j = i+1
        next = data_points[j] if j < len(data_points) else None
        data_points[i]["bearing"] = calculate_bearing(data_points[i], next)
    for i in range(0, len(data_points)):
        j = i+1
        next = data_points[j] if j < len(data_points) else None
        data_points[i]["bearing_rate"] = calculate_bearing_rate(data_points[i], next)
    return data_points

def get_trajectory(points, labels):
    result = []
    for l in labels:
        start = l["start_time"]
        end = l["end_time"]
        val = l["label"]
        df = pd.DataFrame(list(filter(lambda x : x["timestamp"] >= start and  x["timestamp"] <= end, points)))

        if len(df) ==0:
            continue
        r = df[["speed"]]
        speed_min = np.min(r)["speed"]
        speed_max = np.max(r)["speed"]
        speed_mean = np.mean(r)["speed"]
        speed_median = np.median(r)
        speed_std = np.std(r)["speed"]
        r = df[["acceleration"]]
        acceleration_min = np.min(r)["acceleration"]
        acceleration_max = np.max(r)["acceleration"]
        acceleration_mean = np.mean(r)["acceleration"]
        acceleration_median = np.median(r)
        acceleration_std = np.std(r)["acceleration"]
        r = df[["jerk"]]
        jerk_min = np.min(r)["jerk"]
        jerk_max = np.max(r)["jerk"]
        jerk_mean = np.mean(r)["jerk"]
        jerk_median = np.median(r)
        jerk_std = np.std(r)["jerk"]
        r = df[["bearing"]]
        bearing_min = np.min(r)["bearing"]
        bearing_max = np.max(r)["bearing"]
        bearing_mean = np.mean(r)["bearing"]
        bearing_median = np.median(r)
        bearing_std = np.std(r)["bearing"]
        r = df[["bearing_rate"]]
        bearing_rate_min = np.min(r)["bearing_rate"]
        bearing_rate_max = np.max(r)["bearing_rate"]
        bearing_rate_mean = np.mean(r)["bearing_rate"]
        bearing_rate_median = np.median(r)
        bearing_rate_std = np.std(r)["bearing_rate"]
        result.append({
            "label": val, 
            "start_time": start, 
            "end_time": end, 
            "speed_min": speed_min,
            "speed_max": speed_max,
            "speed_mean": speed_mean,
            "speed_median": speed_median,
            "speed_std": speed_std,
            "acceleration_min": acceleration_min,
            "acceleration_max": acceleration_max,
            "acceleration_mean": acceleration_mean,
            "acceleration_median": acceleration_median,
            "acceleration_std": acceleration_std,
            "jerk_min": jerk_min,
            "jerk_max": jerk_max,
            "jerk_mean": jerk_mean,
            "jerk_median": jerk_median,
            "jerk_std": jerk_std,
            "bearing_min": bearing_min,
            "bearing_max": bearing_max,
            "bearing_mean": bearing_mean,
            "bearing_median": bearing_median,
            "bearing_std": bearing_std,
            "bearing_rate_min": bearing_rate_min,
            "bearing_rate_max": bearing_rate_max,
            "bearing_rate_mean": bearing_rate_mean,
            "bearing_rate_median": bearing_rate_median,
            "bearing_rate_std": bearing_rate_std
        })
            #"points": r})
    return result

def get_traj_features(traj):
    pass


class User:
    def __init__(self, name, path):
        self.path = path
        self.name = name
        self.label_file = os.path.join(path, "labels.txt")
        self.has_labels = os.path.exists(self.label_file)
        self.labels = []
        if self.has_labels:
            with open(self.label_file, 'r') as infile:
                for line in infile:
                    splits = line.strip().split()
                    if len(splits) == 5:
                        start_time = datetime.datetime.strptime(splits[0] + " " + splits[1], "%Y/%m/%d %H:%M:%S")
                        end_time = datetime.datetime.strptime(splits[2] + " " + splits[3], "%Y/%m/%d %H:%M:%S")
                        label = splits[4]
                        self.labels.append({"user": self.name, "start_time": start_time, "end_time": end_time, "label": label})

        self.data = []
        traj_path = os.path.join(self.path, "Trajectory")
        for data_file in os.listdir(traj_path):
            print("processing {0}".format(data_file))
            full_data_file = os.path.join(traj_path, data_file)
            with open(full_data_file, 'r') as infile:
                for line in infile:
                    splits = line.strip().split(",")
                    if len(splits) == 7:
                        latitude = float(splits[0])
                        longitude = float(splits[1])
                        zero = int(splits[2])#unused
                        altitude = float(splits[3])
                        day_since = float(splits[4])
                        timestamp = datetime.datetime.strptime(splits[5] + " " + splits[6], "%Y-%m-%d %H:%M:%S")
                        self.data.append({"user": self.name, "day_since": day_since, "timestamp": timestamp, "latitude": latitude, "longitude": longitude, "altitude": altitude})
        print("loaded {0}".format(self.path))

class Dataloader:
    def __init__(self, data_dir=os.path.join("geolife-trajectory-1.3", "Data"), load_portion=1, seed=None):
        random.seed(seed)
        self.labels = []
        self.points = []
        self.traj = []
        for user_folder in os.listdir(data_dir):
            gen = random.uniform(0, 1)
            if gen <= load_portion:
                print("loading {0}/{1}".format(data_dir, user_folder))
                u = User(user_folder, os.path.join(data_dir, user_folder))
                l = u.labels
                p = preprocess(u.data)
                if len(l) == 0:
                    t = []
                else:
                    t = get_trajectory(p, l)
                self.labels += l
                self.points += p
                self.traj += t

    def getDataFrames(self):
        return pd.DataFrame(self.points), pd.DataFrame(self.labels), pd.DataFrame(self.traj)

    def getTrain(self):
        return ([],[])
    def getTest(self):
        return ([],[])
    def getValidate(self):
        return ([],[])
    
#data, lbl = Dataloader().getTrain()
#data, lbl = Dataloader().getTest()
#data, lbl = Dataloader().getValidate()
