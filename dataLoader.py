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
def std_dev(speed_mean,mode_mean,mode_std): 
    if (speed_mean < mode_mean - 2 * mode_std):
        return 1
    elif (speed_mean > mode_mean + 2 * mode_std):
        return 1
    else:
        return 0

def remove_noise(data):
    mode_mean = data.groupby(['label'])['speed_mean'].mean().to_frame(name='mode_mean').reset_index()
    data_mean = data.merge(mode_mean,left_on='label', right_on='label')
    mode_std = data.groupby(['label'])['speed_mean'].std().to_frame(name='mode_std').reset_index()
    stdev = data_mean.merge(mode_std, left_on='label', right_on='label')
    stdev["to_remove"] = stdev.apply(lambda x: std_dev(x["speed_mean"], x["mode_mean"], x["mode_std"]), axis = 1)
    result = stdev[stdev['to_remove'] != 1]
    result.pop('mode_mean')
    result.pop('mode_std')
    result.pop('to_remove')
    return result

class Dataloader:
    def __init__(self, select_features=None, normalization=False, noise_removal=False):
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
        
        #drop rows with label not walk, bike, bus, taxi, car, subway, train
        data = data[data['label'].map(lambda x: self.classes[x]) <=4]
        
        if noise_removal:
            data = remove_noise(data)
            
        self.Y = data["label"].map(lambda x: self.classes[x])
        data.pop("label")

        self.X = data
        if normalization:
            self.X = pd.DataFrame(preprocessing.MinMaxScaler().fit_transform(self.X.values))
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y, test_size=0.1)
        self.X_train, self.X_val, self.Y_train, self.Y_val = train_test_split(self.X_train, self.Y_train, test_size=0.1)

    def getX(self):
        return (self.X)
    def getTrain(self):
        return (self.X_train,self.Y_train)
    def getTest(self):
        return (self.X_test,self.Y_test)
    def getValidate(self):
        return (self.X_val, self.Y_val)
    def getClasses(self):
        return self.classes

