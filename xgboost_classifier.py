# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 19:25:42 2019

@author: User
"""

import xgboost as xgb
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV,ShuffleSplit,train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import preprocessing


xgb_base_model = xgb.XGBClassifier(random_state = 8)

print('Parameters currently in use:\n')
print(xgb_base_model.get_params())

classes = ["taxi","walk", "bike", "bus", "car", "subway", "train", "airplane", "boat", "run", "motorcycle"]

category_codes = {
    'taxi': 0,
    'walk': 1,
    'bike': 2,
    'bus': 3,
    'car': 4,
    'subway': 5,
    'train': 6,
    'airplane': 7,
    'boat': 8,
    'run': 9,
    'motorcycle': 10
}

#_, _, traj = Dataloader(load_portion=0.03).getDataFrames()
#print(traj)

traj=pd.read_excel("traj.xlsx")

traj = traj.replace({'labels':category_codes})

labels = traj[['label']]
del traj['label']
del traj['start_time']
del traj['end_time']

X_train, X_test, y_train, y_test = train_test_split(traj,labels,test_size = 0.2, random_state = 0)

print(X_train.shape)
print(y_train.shape)

xgb_base_model = xgb.XGBClassifier(random_state = 8)


print('Parameters currently in use:\n')
print(xgb_base_model.get_params())


## Randomized Search Cross Validation

# n_estimators
n_estimators = [200, 800]

# max_features
max_features = ['auto', 'sqrt']

# max_depth
max_depth = [10, 40]

# min_samples_split
min_samples_split = [10, 30, 50]

# min_samples_leaf
min_samples_leaf = [1, 2, 4]

# learning rate
learning_rate = [.1, .5]

# subsample
subsample = [.5, 1.]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'learning_rate': learning_rate,
               'subsample': subsample}

print(random_grid)

# First create the base model to tune
xgb_base_model = xgb.XGBClassifier(random_state = 8)

# Definition of the random search
random_search = RandomizedSearchCV(estimator=xgb_base_model,
                                   param_distributions=random_grid,
                                   n_iter=50,
                                   scoring='accuracy',
                                   cv=3, 
                                   verbose=1, 
                                   random_state=8)

# Fit the random search model
random_search.fit(X_train, y_train)

print("The best hyperparameters from Random Search are:")
print(random_search.best_params_)
print("")
print("The mean accuracy of a model with these hyperparameters is:")
print(random_search.best_score_)

## Grid Search Cross Validation
# Create the parameter grid based on the results of random search 
max_depth = [5, 10, 15]
max_features = ['sqrt']
min_samples_leaf = [2]
min_samples_split = [50, 100]
n_estimators = [800]
learning_rate = [.1, .5]
subsample = [1.]

param_grid = {
    'max_depth': max_depth,
    'max_features': max_features,
    'min_samples_leaf': min_samples_leaf,
    'min_samples_split': min_samples_split,
    'n_estimators': n_estimators,
    'learning_rate': learning_rate,
    'subsample': subsample

}

# Create a base model
gbc = xgb.XGBClassifier(random_state=8)

# Manually create the splits in CV in order to be able to fix a random_state (GridSearchCV doesn't have that argument)
cv_sets = ShuffleSplit(n_splits = 3, test_size = .33, random_state = 8)

# Instantiate the grid search model
grid_search = GridSearchCV(estimator=gbc, 
                           param_grid=param_grid,
                           scoring='accuracy',
                           cv=cv_sets,
                           verbose=1)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)