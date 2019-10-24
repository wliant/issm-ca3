import numpy as np
import sklearn.metrics as metrics
import matplotlib.pyplot as plt


from tensorflow.keras.callbacks import ModelCheckpoint,CSVLogger,LearningRateScheduler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import add
from tensorflow.keras import regularizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import tensorflow as tf
import pandas as pd
import os
from dataLoader import Dataloader

output_folder = 'output'
classes = ["walk", "bike", "bus", "car", "subway", "train", "airplane", "boat", "run", "motorcycle"]
batch_size = 32
IMG_SIZE = 300

def implt(img):
    plt.figure()
    plt.imshow(img)
    plt.axis('off')

plt.style.use('ggplot')     # if want to use the default style, set 'classic'
plt.rcParams['ytick.right']     = True
plt.rcParams['ytick.labelright']= True
plt.rcParams['ytick.left']      = False
plt.rcParams['ytick.labelleft'] = False
plt.rcParams['font.family']     = 'Arial'
modelname = 'pre-1'
seed = 7
np.random.seed(seed)

# .............................................................................
_, _, traj = Dataloader(load_portion=0.03).getDataFrames()
print(traj)
#https://www.tensorflow.org/tutorials/load_data/pandas_dataframe
enc = OneHotEncoder(categories=[classes], handle_unknown='ignore')
tofit = traj[['label']]
labels = enc.fit_transform(tofit).toarray()
print(labels.shape)

traj.pop("label")
traj.pop("start_time")
traj.pop("end_time")
dat = tf.convert_to_tensor(traj.values)
lbl = tf.convert_to_tensor(pd.DataFrame(labels).values)
dataset = tf.data.Dataset.from_tensor_slices((dat, lbl))

filepath        = os.path.join(output_folder, modelname + ".hdf5")
checkpoint      = ModelCheckpoint(filepath, 
                                  monitor='val_acc', 
                                  verbose=0, 
                                  save_best_only=True, 
                                  mode='max')
                            # Log the epoch detail into csv
csv_logger      = CSVLogger(os.path.join(output_folder, modelname +'.csv'))
callbacks_list  = [checkpoint,csv_logger]

#---- model creation code
def get_compiled_model():
  model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(len(classes), activation='softmax')
  ])

  model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
  return model
def createModel():
    i = Input(shape=(IMG_SIZE,IMG_SIZE,3))
    layer = Conv2D(32, kernel_size = (3,3), activation='relu')(i)
    layer = MaxPooling2D(pool_size=(2,2))(layer)
    layer = BatchNormalization()(layer)
    layer = Conv2D(64, kernel_size=(3,3), activation='relu')(layer)
    layer = MaxPooling2D(pool_size=(2,2))(layer)
    layer = BatchNormalization()(layer)
    layer = Conv2D(64, kernel_size=(3,3), activation='relu')(layer)
    layer = MaxPooling2D(pool_size=(2,2))(layer)
    layer = BatchNormalization()(layer)
    layer = Conv2D(96, kernel_size=(3,3), activation='relu')(layer)
    layer = MaxPooling2D(pool_size=(2,2))(layer)
    layer = BatchNormalization()(layer)
    layer = Conv2D(32, kernel_size=(3,3), activation='relu')(layer)
    layer = MaxPooling2D(pool_size=(2,2))(layer)
    layer = BatchNormalization()(layer)
    layer = Dropout(0.2)(layer)
    layer = Flatten()(layer)
    layer = Dense(128, activation='relu')(layer)
    layer = Dense(128, activation='relu')(layer)
    layer = Dense(128, activation='relu')(layer)
    layer = Dense(len(classes), activation = 'softmax')(layer)
    model = Model(inputs=i, outputs=layer)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
  
  # define model
#model = createModel()
model = get_compiled_model()
#model.summary()

# fit model
#model.fit_generator(train_it, validation_data=val_it,epochs=50,callbacks=callbacks_list)
model.fit(dataset, epochs=15)

from tensorflow.keras.utils import plot_model
model_file = os.path.join(output_folder, modelname + "_model.png")
plot_model(model, 
           to_file=model_file, 
           show_shapes=True, 
           show_layer_names=False,
           rankdir='TB')