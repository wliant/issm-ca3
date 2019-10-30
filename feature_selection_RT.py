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
from sklearn.metrics import accuracy_score



x_train,y_train = Dataloader().getTrain()
x_test,y_test = Dataloader().getTest()

y_train = to_categorical(y_train)
x_train.pop("start_time")
x_train.pop("end_time")
x_test.pop("start_time")
x_test.pop("end_time")
print(x_train)


print(x_train.shape)
print(y_train.shape)

clf = RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1)


# Apply The Full Featured Classifier To The Test Data
clf.fit(x_train,y_train)

sel = SelectFromModel(clf)
sel.fit(x_train, y_train)
selected_feat= x_train.columns[(sel.get_support())]
len(selected_feat)

print(selected_feat)

X_important_train = sel.transform(x_train)
X_important_test = sel.transform(x_test)

# Create a new random forest classifier for the most important features
#clf_important = RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1)

# Train the new classifier on the new dataset containing the most important features
#clf_important.fit(X_important_train, y_train)

#y_pred = clf.predict(x_test)

# View The Accuracy Of Our Full Feature (4 Features) Model
#print(accuracy_score(y_test,  y_pred.argmax(axis=1)))

# Apply The Full Featured Classifier To The Test Data
#y_important_pred = clf_important.predict(X_important_test)

# View The Accuracy Of Our Limited Feature (2 Features) Model
#print(accuracy_score(y_test, y_important_pred.argmax(axis=1)))

def getSelectedFeature():
    return selected_feat

def getNewXData():
    return (X_important_train,X_important_test)