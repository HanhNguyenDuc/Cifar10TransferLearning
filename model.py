from keras.layers import *
from keras.models import *
from keras.datasets import cifar10
from keras.optimizers import SGD
import numpy as np
from keras.callbacks import *
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.preprocessing.image import img_to_array

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
IMAGE_SIZE = X_train.shape[1:]


def model_transfer():
    input_ = Input(shape = IMAGE_SIZE)
    upsamp_ = UpSampling2D(size = (2, 2))(input_)
    init_model = ResNet50(include_top = False, weights = 'imagenet', classes = 10)
    flatten_ = GlobalAveragePooling2D()(init_model.output)
    dense_10 = Dense(1024)(flatten_)
#     drop_ = Dropout(0.25)(dense_10)
    
    soft_max = Dense(10, activation = 'softmax')(dense_10)
    return Model(inputs = [init_model.input], outputs = [soft_max])
    

def my_learning_rate(epoch, lrate):
    return lrate

model = model_transfer()
print(model.summary())
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

X_train = X_train / 255
X_test = X_test / 255

endp = int(len(X_train) * 0.9)


print(X_train.shape)

model.fit(X_train, y_train, epochs = 5)

loss, score = model.evaluate(X_test, y_test)

print(loss, score)


