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
import cv2
from keras.callbacks import ModelCheckpoint

(X_train_, y_train_), (X_test, y_test) = cifar10.load_data()
IMAGE_SIZE = X_train_.shape[1:]

datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True)


def model_transfer():
    init_model = VGG16(include_top = True,weights = None, input_shape = IMAGE_SIZE)
       
    return init_model

  
model = model_transfer()
print(model.summary())
adam = Adam(lr = 1e-4)
model.compile(optimizer = adam, loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])


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

filepath = "weights.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor = 'val_acc', verbose = 1, save_best_only = True, mode = 'max')


def resizeimg(img):
    return cv2.resize(img, dsize = (224, 224), interpolation = cv2.INTER_CUBIC)
  
# X_train = np.stack(map(resizeimg, X_train))
# X_test = np.stack(map(resizeimg, X_test))
# X_val = np.stack(map(resizeimg, X_val))

print(X_train.shape)

datagen.fit(X_train)
model.fit_generator(datagen.flow(X_train, y_train, batch_size = 32), validation_data = (X_val, y_val), epochs = 100, steps_per_epoch = len(X_train) / 32,
                   callbacks = [checkpoint], verbose = 1)

loss, score = model.evaluate(X_test, y_test)

print(loss, score)

#Change 1: Trained with model vgg16 but didn't load weight, init learning rate by 1e-4.
#Change 2: Used callbacks to save best model and weight. Test acc = .83 with 25 epochs
#Change 3: train with 100 epochs





