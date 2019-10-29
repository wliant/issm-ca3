#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 20:05:17 2019

@author: tangmeng
"""

from dataLoader import Dataloader
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

from tensorflow.python.keras.utils import to_categorical


x_train,y_train = Dataloader().getTrain()


y_train = to_categorical(y_train)
x_train.pop("start_time")
x_train.pop("end_time")

print(x_train)


print(x_train.shape)
print(y_train.shape)

sel = SelectFromModel(RandomForestClassifier(n_estimators = 100))
sel.fit(x_train, y_train)
selected_feat= x_train.columns[(sel.get_support())]
len(selected_feat)

print(selected_feat)

def getSelectedFeature(self):
    return selected_feat