import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
columns = [
            "label", 
            "point_count",
            "start_time",
            "end_time",
            "speed_min",
            "speed_max",
            "speed_mean",
            "speed_median",
            "speed_std",
            "acceleration_min",
            "acceleration_max",
            "acceleration_mean",
            "acceleration_median",
            "acceleration_std",
            "jerk_min",
            "jerk_max",
            "jerk_mean",
            "jerk_median",
            "jerk_std",
            "bearing_min",
            "bearing_max",
            "bearing_mean",
            "bearing_median",
            "bearing_std",
            "bearing_rate_min",
            "bearing_rate_max",
            "bearing_rate_mean",
            "bearing_rate_median",
            "bearing_rate_std",
            "bearing_rate_rate_min",
            "bearing_rate_rate_max",
            "bearing_rate_rate_mean",
            "bearing_rate_rate_median",
            "bearing_rate_rate_std"
        ]


class Dataloader:
    def __init__(self, select_features=None, normalization=False):
        self.classes = {
            "walk":0,
            "bike":1,
            "bus":2,
            "taxi": 3,
            "car":3, 
            "subway":4, 
            "train":4, 
            "airplane":6, 
            "boat":7, 
            "run":8, 
            "motorcycle":10
        }
        data = pd.read_csv("traj.csv", index_col =False, header=None,
                names=columns)
        if select_features is not None:
            data = data[["label"] + select_features]
        #print("before filtering: {0}".format(data.shape))
        #drop rows with label not walk, bike, bus, taxi, car, subway, train
        data = data[data['label'].map(lambda x: self.classes[x]) <=4]

        #print("after filtering: {0}".format(data.shape))
        self.Y = data["label"].map(lambda x: self.classes[x])
        data.pop("label")

        self.X = data
        if normalization:
            self.X = pd.DataFrame(preprocessing.MinMaxScaler().fit_transform(self.X.values))
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y, test_size=0.1)
        self.X_train, self.X_val, self.Y_train, self.Y_val = train_test_split(self.X_train, self.Y_train, test_size=0.1)

    def getTrain(self):
        return (self.X_train,self.Y_train)
    def getTest(self):
        return (self.X_test,self.Y_test)
    def getValidate(self):
        return (self.X_val, self.Y_val)
    def getClasses(self):
        return self.classes