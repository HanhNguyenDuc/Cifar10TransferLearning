from keras.layers import *
from keras.models import *
from keras.datasets import cifar10
from keras.optimizers import SGD, Adam
import numpy as np
from keras.callbacks import *
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import *
from keras.preprocessing.image import img_to_array, ImageDataGenerator
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedShuffleSplit
from keras.callbacks import ModelCheckpoint

(X_train_, y_train_), (X_test, y_test) = cifar10.load_data()
IMAGE_SIZE = X_train_.shape[1:]

datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True)


def model_transfer(not_froze = 0):
    input_ = Input(shape = IMAGE_SIZE)
    init_model = VGG16(include_top = False, weights = 'imagenet')
    for layer in init_model.layers:
      layer.trainable = False
          
    resnet_layers = init_model(input_)
    flatten_ = Flatten()(resnet_layers)
    norm_ = BatchNormalization()(flatten_)
    dense_10 = Dense(256)(norm_)
    drop_ = Dropout(0.25)(dense_10)
    dense_11 = Dense(256)(drop_)
    norm_2 = BatchNormalization()(dense_11)
    drop_2 = Dropout(0.25)(norm_2)
    
    
    soft_max = Dense(10, activation = 'softmax')(drop_2)
    return Model(inputs = [input_], outputs = [soft_max])
    

def my_learning_rate(epoch, lrate):
    return lrate

  
model = model_transfer()
print(model.summary())
adam = Adam(lr = 1e-6)
model.compile(optimizer = adam, loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

checkpoint = ModelCheckpoint('weight.hdf5', monitor = 'val_acc', verbose = 1, mode = 'max', save_best_only = True)

sss = StratifiedShuffleSplit(n_splits = 5, test_size = 0.1, random_state = 0)

X_train = []
y_train = []
X_val = []
y_val = []

for train_index, val_index in sss.split(X_train_, y_train_):
    X_train, X_val = X_train_[train_index], X_train_[val_index]
    y_train, y_val = y_train_[train_index], y_train_[val_index]
    print(X_train.shape)
    print(train_index.shape)

X_train = preprocess_input(X_train)
X_val = preprocess_input(X_val)
X_test = preprocess_input(X_test)


print(X_train.shape)

datagen.fit(X_train)
model.fit_generator(datagen.flow(X_train, y_train, batch_size = 32), validation_data = (X_val, y_val), epochs = 100, steps_per_epoch = len(X_train) / 32,
                   callbacks = [checkpoint])

loss, score = model.evaluate(X_test, y_test)

print(loss, score)

#Change 1: Learning rate = 1e-4, no trainable layer in convnet, 2 dense with 256 unit, after each dense layer are batchnorm and dropout .25, 25 epochs
 #Acc = .65
#Change 2: Learning rate = 1e-5, train with 25 epochs, acc = .645

#Change 3: learning rate = 1e-5, train with 100 epochs


