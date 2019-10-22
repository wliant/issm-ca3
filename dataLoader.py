import os
import datetime
import random
import pandas as pd 
import numpy as np
from vincenty import vincenty

THRESHOLD = 30

def calculate_speed(current, next):
    if next is None:
        return 0
    distance = 1000*vincenty((current["latitude"],current["longitude"]),(next["latitude"],next["longitude"]))
    time = (next["timestamp"]-current["timestamp"]).seconds
    if time <= 0:
        time = 1
    if time > THRESHOLD:
        return 0
    #print("{0} {1} {2}".format(current["timestamp"], next["timestamp"], next["timestamp"]-current["timestamp"]))
    #print("{0} {1}".format(distance, time))
    return distance /time
def calculate_acceleration(current, next):
    if next is None:
        return 0
    speed_diff = next["speed"] - current["speed"]
    time = (next["timestamp"]-current["timestamp"]).seconds
    if time <= 0:
        time = 1
    if time > THRESHOLD:
        return 0
    return speed_diff / time

def calculate_jerk(current, next):
    if next is None:
        return 0
    acc_diff = next["acceleration"] - current["acceleration"]
    time = (next["timestamp"] - current["timestamp"]).seconds
    if time <= 0:
        time = 1
    if time > THRESHOLD:
        return 0
    return acc_diff / time
def calculate_bearing(current, next):
    if next is None:
        return 0
    time = (next["timestamp"] - current["timestamp"]).seconds
    if time <= 0:
        time = 1
    if time > THRESHOLD:
        return 0
    xx = np.cos(current["latitude"] * np.pi/180) * np.sin(next["latitude"] * np.pi/180) - np.sin(current["latitude"] * np.pi/180) * np.cos(next["latitude"] * np.pi/180) * np.cos((next["longitude"] - current["longitude"]) * np.pi/180)
    yy = np.sin(next["longitude"] - current["latitude"] * np.pi/180) * np.cos(next["latitude"] * np.pi/180)
    return np.arctan2(yy,xx) * 180/np.pi
def y(x):
    try:
        yy = np.sin((x[8] - x[0]) * np.pi/180) *np.cos( x[7]* np.pi/180)
    except:
        yy= np.nan
    return yy
def x(x):
    try:
        xx = np.cos(x[0] * np.pi/180) *np.sin(x[7]* np.pi/180) - np.sin(x[0]* np.pi/180) * np.cos(x[7]* np.pi/180)*np.cos((x[8]-x[1])* np.pi/180)
    except:
        xx = np.nan
    return xx
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
    return data_points

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
            full_data_file = os.path.join(traj_path, data_file)
            with open(full_data_file, 'r') as infile:
                for line in infile:
                    splits = line.strip().split(",")
                    if len(splits) == 7:
                        latitude = float(splits[0])
                        longitude = float(splits[1])
                        zero = int(splits[2])#unused
                        altitude = float(splits[3])
                        day_since = float(splits[4])#unused 
                        timestamp = datetime.datetime.strptime(splits[5] + " " + splits[6], "%Y-%m-%d %H:%M:%S")
                        self.data.append({"user": self.name, "timestamp": timestamp, "latitude": latitude, "longitude": longitude, "altitude": altitude})
        print("loaded {0}".format(self.path))

class Dataloader:
    def __init__(self, data_dir=os.path.join("geolife-trajectory-1.3", "Data"), load_portion=1, seed=None):
        random.seed(seed)
        self.labels = []
        self.points = []
        for user_folder in os.listdir(data_dir):
            gen = random.uniform(0, 1)
            if gen <= load_portion:
                u = User(user_folder, os.path.join(data_dir, user_folder))
                self.labels += u.labels
                self.points += preprocess(u.data)
                
    def getDataFrames(self):
        return pd.DataFrame(self.points), pd.DataFrame(self.labels)

points, labels = Dataloader(load_portion=0.02).getDataFrames()
print(points)