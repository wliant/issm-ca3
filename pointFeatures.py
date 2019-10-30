from dataLoader import Dataloader
from vincenty import vincenty

points, labels, traj = Dataloader(load_portion=0.02).getDataFrames()
print(points)
print(labels)
print(traj)

data, lbl = Dataloader().getTrain()
data, lbl = Dataloader().getTest()
data, lbl = Dataloader().getValidate()