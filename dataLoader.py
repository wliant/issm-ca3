import os
import datetime
import random
import pandas
data_folder = os.path.join("geolife-trajectory-1.3", "Data")

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
                        self.labels.append({"start_time": start_time, "end_time": end_time, "label": label})
        
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
                        self.data.append({ "timestamp": timestamp, "latitude": latitude, "longitude": longitude, "altitude": altitude})
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
                self.points += u.data
                
    def getDataFrames(self):
        
dl = Dataloader(load_portion=0.05)
#users = []                    
#for user_folder in os.listdir(data_folder):
    #print(user_folder)
#    users.append(User(os.path.join(data_folder, user_folder)))
    
#print("load complete")