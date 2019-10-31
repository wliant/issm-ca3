from dataLoader import Dataloader
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from tensorflow.python.keras.utils import to_categorical



x_train,y_train = Dataloader().getTrain()
x_test,y_test = Dataloader().getTest()

y_train = to_categorical(y_train)
x_train.pop("start_time")
x_train.pop("end_time")
x_test.pop("start_time")
x_test.pop("end_time")

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


def getSelectedFeature():
    return selected_feat
