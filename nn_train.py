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
from tensorflow.keras.utils import to_categorical

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
dl = Dataloader()
x_train, y_train = dl.getTrain()
x_test, y_test = dl.getTest()
print(y_train)
#enc = OneHotEncoder(categories=[classes],handle_unknown='ignore',drop=[0])
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
x_train.pop("start_time")
x_test.pop("start_time")
x_train.pop("end_time")
x_test.pop("end_time")
dat = tf.convert_to_tensor(x_train)
print(dat.shape)
lbl = tf.convert_to_tensor(y_train)
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


def createModel():
    i = Input(shape=(24,1))
    layer = Conv1D(32, kernel_size = 3, activation='relu')(i)
    layer = MaxPooling1D(pool_size=2)(layer)
    layer = BatchNormalization()(layer)
    layer = Conv1D(64, kernel_size = 3, activation='relu')(i)
    layer = MaxPooling1D(pool_size=2)(layer)
    layer = BatchNormalization()(layer)
    layer = Conv1D(64, kernel_size = 3, activation='relu')(i)
    layer = MaxPooling1D(pool_size=2)(layer)
    layer = BatchNormalization()(layer)
    layer = Dropout(0.2)(layer)
    layer = Flatten()(layer)
    layer = Dense(128, activation='relu')(layer)
    layer = Dense(128, activation='relu')(layer)
    layer = Dense(len(classes), activation = 'softmax')(layer)
    model = Model(inputs=i, outputs=layer)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
  
  # define model
#model = createModel()
model = createModel()
model.summary()
from tensorflow.keras.utils import plot_model
model_file = os.path.join(output_folder, modelname + "_model.png")
plot_model(model, 
           to_file=model_file, 
           show_shapes=True, 
           show_layer_names=False,
           rankdir='TB')
# fit model
#model.fit_generator(train_it, validation_data=val_it,epochs=50,callbacks=callbacks_list)
model.fit(dataset, epochs=15, steps_per_epoch=5)

