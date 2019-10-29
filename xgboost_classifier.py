# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 19:25:42 2019

@author: User
"""

import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV,ShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from dataLoader import Dataloader

import matplotlib.pyplot as plt
import seaborn as sns

#Load Data
dl = Dataloader()
X_train, y_train = dl.getTrain()

X_test, y_test = dl.getTest()

X_validate, y_validate = dl.getValidate()

print(X_train.shape)
print(y_train.shape)

classes = dl.getClasses()

inv_map = {v: k for k, v in classes.items()}

print(classes)

#Base Model
xgb_base_model = xgb.XGBClassifier(random_state = 8)


print('Parameters currently in use:\n')
print(xgb_base_model.get_params())


## Randomized Search Parameters
n_estimators = [200, 800]
max_features = ['auto', 'sqrt']
max_depth = [10, 40]
min_samples_split = [10, 30, 50]
min_samples_leaf = [1, 2, 4]
learning_rate = [.1, .5]
subsample = [.5, 1.]
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'learning_rate': learning_rate,
               'subsample': subsample}

print(random_grid)


# Random Search
random_search = RandomizedSearchCV(estimator=xgb_base_model,
                                   param_distributions=random_grid,
                                   n_iter=30,
                                   scoring='accuracy',
                                   cv=3, 
                                   verbose=1, 
                                   random_state=8)
random_search.fit(X_validate, y_validate)

print("The best hyperparameters from Random Search are:")
print(random_search.best_params_)
print("")
print("The mean accuracy of a model with these hyperparameters is:")
print(random_search.best_score_)

## Grid Search 
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

# Manually create the splits in CV to fix a random_state
cv_sets = ShuffleSplit(n_splits = 1, test_size = .33, random_state = 8)

# Grid Search
grid_search = GridSearchCV(estimator=xgb_base_model, 
                           param_grid=param_grid,
                           scoring='accuracy',
                           cv=cv_sets,
                           verbose=1)
grid_search.fit(X_validate, y_validate)
print("The best hyperparameters from Grid Search are:")
print(grid_search.best_params_)
print("")
print("The mean accuracy of a model with these hyperparameters is:")
print(grid_search.best_score_)

best_xgb= grid_search.best_estimator_

best_xgb

## Model fit and performance

best_xgb.fit(X_train, y_train)

gbc_pred = best_xgb.predict(X_test)

# Training accuracy
print("The training accuracy is: ")
print(accuracy_score(y_train, best_xgb.predict(X_train)))

# Test accuracy
print("The test accuracy is: ")
print(accuracy_score(y_test, gbc_pred))


# Classification report
print("Classification report")
print(classification_report(y_test,gbc_pred))

conf_matrix = confusion_matrix(y_test, gbc_pred)

labels=inv_map.values()
    
print(conf_matrix)
plt.figure(figsize=(12.8,6))
sns.heatmap(conf_matrix, 
            annot=True,
            xticklabels=labels, 
            yticklabels=labels,
            cmap="Blues")
plt.ylabel('Predicted')
plt.xlabel('Actual')
plt.title('Confusion matrix')
plt.show()


best_xgb.fit(X_train, y_train)
accuracy_score(y_test, best_xgb.predict(X_test))