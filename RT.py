#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 18:57:46 2019

@author: tangmeng
"""


import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from pprint import pprint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import tensorflow as tf

import feature_selection_RT as fsrt

from tensorflow.python.keras.utils import to_categorical


from dataLoader import Dataloader

x_important_train,x_important_test = fsrt.getNewXData()
x_train,y_train = Dataloader().getTrain()
x_test,y_test = Dataloader().getTest()

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
#x_train.pop("start_time")
#x_test.pop("start_time")
#x_train.pop("end_time")
#x_test.pop("end_time")

x_important_train = np.expand_dims(x_important_train,axis=2)
x_important_test = np.expand_dims(x_important_test,axis=2)
print(x_important_train.shape)
print(y_train.shape)

nsamples, nx, ny = x_important_train.shape
x_important_train = x_important_train.reshape((nsamples,nx*ny))

nsamples, nx, ny = x_important_test.shape
x_important_test = x_important_test.reshape((nsamples,nx*ny))


rf_0 = RandomForestClassifier(random_state = 8)

print('Parameters currently in use:\n')
pprint(rf_0.get_params())

# n_estimators
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 1000, num = 5)]

# max_features
max_features = ['auto', 'sqrt']

# max_depth
max_depth = [int(x) for x in np.linspace(20, 100, num = 5)]
max_depth.append(None)

# min_samples_split
min_samples_split = [2, 5, 10]

# min_samples_leaf
min_samples_leaf = [1, 2, 4]

# bootstrap
bootstrap = [True, False]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

pprint(random_grid)

# First create the base model to tune
rfc = RandomForestClassifier(random_state=8)

# Definition of the random search
random_search = RandomizedSearchCV(estimator=rfc,
                                   param_distributions=random_grid,
                                   n_iter=50,
                                   scoring='accuracy',
                                   cv=3, 
                                   verbose=1, 
                                   random_state=8)

# Fit the random search model
random_search.fit(x_important_train, y_train)

print("The best hyperparameters from Random Search are:")
print(random_search.best_params_)
print("")
print("The mean accuracy of a model with these hyperparameters is:")
print(random_search.best_score_)

# Create the parameter grid based on the results of random search 
bootstrap = [False]
max_depth = [30, 40, 50]
max_features = ['sqrt']
min_samples_leaf = [1, 2, 4]
min_samples_split = [5, 10, 15]
n_estimators = [800]

param_grid = {
    'bootstrap': bootstrap,
    'max_depth': max_depth,
    'max_features': max_features,
    'min_samples_leaf': min_samples_leaf,
    'min_samples_split': min_samples_split,
    'n_estimators': n_estimators
}

# Create a base model
rfc = RandomForestClassifier(random_state=8)

# Manually create the splits in CV in order to be able to fix a random_state (GridSearchCV doesn't have that argument)
cv_sets = ShuffleSplit(n_splits = 3, test_size = .33, random_state = 8)

# Instantiate the grid search model
grid_search = GridSearchCV(estimator=rfc, 
                           param_grid=param_grid,
                           scoring='accuracy',
                           cv=cv_sets,
                           verbose=1)

# Fit the grid search to the data
grid_search.fit(x_important_train, y_train)

print("The best hyperparameters from Grid Search are:")
print(grid_search.best_params_)
print("The mean accuracy of a model with these hyperparameters is:")
print(grid_search.best_score_)

best_rfc = grid_search.best_estimator_
best_rfc.fit(x_important_train, y_train)
rfc_pred = best_rfc.predict(x_important_test)

# Training accuracy
print("The training accuracy is: ")
print(accuracy_score(x_important_train, best_rfc.predict(x_important_train)))

# Test accuracy
print("The test accuracy is: ")
print(accuracy_score(y_test, rfc_pred))

# Classification report
print("Classification report")
print(classification_report(y_test,rfc_pred))




