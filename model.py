from keras.layers import *
from keras.models import *
from keras.datasets import cifar10
from keras.optimizers import SGD
import numpy as np
from keras.callbacks import *
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import *
from keras.preprocessing.image import img_to_array

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
IMAGE_SIZE = X_train.shape[1:]


def model_transfer():
    input_ = Input(shape = IMAGE_SIZE)
#     upsamp_ = UpSampling2D(size = (2, 2))(input_)
    init_model = VGG16(include_top = False, weights = 'imagenet')
    init_model.summary()
    for layer in init_model.layers[:-3]:
        layer.trainable = False
    
    resnet_layers = init_model(input_)
    flatten_ = Flatten()(resnet_layers)
    norm_ = BatchNormalization()(flatten_)
    dense_10 = Dense(128)(norm_)
    drop_ = Dropout(0.5)(dense_10)
    
    soft_max = Dense(10, activation = 'softmax')(drop_)
    return Model(inputs = [input_], outputs = [soft_max])
    

def my_learning_rate(epoch, lrate):
    return lrate

model = model_transfer()
print(model.summary())
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

endp = int(len(X_train) * 0.9)

X_val = X_train[endp:]
y_val = y_train[endp:]

X_train = X_train[:endp]
y_train = y_train[:endp]

X_train = preprocess_input(X_train)
X_val = preprocess_input(X_val)
X_test = preprocess_input(X_test)

endp = int(len(X_train) * 0.9)


print(X_train.shape)

model.fit(X_train, y_train, epochs = 50, validation_data = (X_val, y_val))

loss, score = model.evaluate(X_test, y_test)

print(loss, score)


