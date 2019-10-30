#Import Packages
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix 
import warnings
warnings.filterwarnings('ignore')
from dataLoader import Dataloader
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

#Load Data
dl = Dataloader()

X = dl.getX()
X_train, y_train = dl.getTrain()
X_test, y_test = dl.getTest()

print(X_train.shape)
print(y_train.shape)

n_features = 33 #to edit if there's any features that are dropped

classes = dl.getClasses()
inv_map = {v: k for k, v in classes.items()}

print(classes)

#Create Decision Tree Model
dt = DecisionTreeClassifier(criterion='entropy',random_state=0)
dt.fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(dt.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(dt.score(X_test, y_test)))

#Visualization of DT before pruning (dot: graph is too large for cairo-renderer bitmaps)
#Feature Importance before pruning

print("Feature importances:\n{}".format(dt.feature_importances_))

def plot_feature_importances(model):
    traj_features = [x for i,x in enumerate(X.columns) if i!=n_features]
    plt.figure(figsize=(8,6))
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), traj_features)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.ylim(-1, n_features)
plot_feature_importances(dt)
plt.savefig('feature_importance')

#Pruning DT
dt_pruned = DecisionTreeClassifier(criterion='entropy',max_depth=6, random_state=0)
dt_pruned.fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(dt_pruned.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(dt_pruned.score(X_test, y_test)))

y_pred = dt_pruned.predict(X_test)

print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred)) 

#Visualization of DT after pruning
dot_data = StringIO()
export_graphviz(dt_pruned, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = X.columns,class_names=["taxi","walk", "bike", "bus", "car", "subway", "train", "airplane", "boat", "run", "motorcycle"])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('DT_pruned.png')
Image(graph.create_png())

#Feature Importance

print("Feature importances:\n{}".format(dt_pruned.feature_importances_))

def plot_feature_importances(model):
    traj_features = [x for i,x in enumerate(X.columns) if i!=n_features]
    plt.figure(figsize=(8,6))
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), traj_features)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.ylim(-1, n_features)
plot_feature_importances(dt_pruned)
plt.savefig('feature_importance_pruned')