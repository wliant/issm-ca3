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
import matplotlib.pyplot as plt
import seaborn as sns

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

traj['label_code']=traj['label']
traj.replace({'label_code':category_codes})

labels = traj[['label_code']]
dup = traj.copy()
del dup['label']
del dup['start_time']
del dup['end_time']
del dup['label_code']


X_train, X_test, y_train, y_test = train_test_split(dup,labels,test_size = 0.2, random_state = 0)

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


print("The best hyperparameters from Grid Search are:")
print(grid_search.best_params_)
print("")
print("The mean accuracy of a model with these hyperparameters is:")
print(grid_search.best_score_)

best_gbc = grid_search.best_estimator_

best_gbc

## Model fit and performance

best_gbc.fit(X_train, y_train)

gbc_pred = best_gbc.predict(X_test)

# Training accuracy
print("The training accuracy is: ")
print(accuracy_score(y_train, best_gbc.predict(X_train)))

# Test accuracy
print("The test accuracy is: ")
print(accuracy_score(y_test, gbc_pred))


# Classification report
print("Classification report")
print(classification_report(y_test,gbc_pred))

aux_df = traj[['label','label_code']].drop_duplicates().sort_values('label_code')

print(aux_df)
conf_matrix = confusion_matrix(y_test, gbc_pred)
plt.figure(figsize=(12.8,6))
sns.heatmap(conf_matrix, 
            annot=True,
            xticklabels=aux_df['label'].values, 
            yticklabels=aux_df['label'].values,
            cmap="Blues")
plt.ylabel('Predicted')
plt.xlabel('Actual')
plt.title('Confusion matrix')
plt.show()

base_model = xgb.XGBClassifier(random_state = 8)
base_model.fit(X_train, y_train)
accuracy_score(y_test, base_model.predict(X_test))

best_gbc.fit(X_train, y_train)
accuracy_score(y_test, best_gbc.predict(X_test))