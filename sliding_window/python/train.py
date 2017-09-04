import scipy.io
import keras
import numpy as np
from sklearn.metrics import confusion_matrix

TRAIN = 1
TEST = 3
VAL = 2


INPUT_SHAPE = (32,32,1)

# Load data
data = scipy.io.loadmat("data/imdb.mat")
data = data['imdb'][0][0][0][0][0]
ids = data[0].flatten()
images = data[1]
labels = data[2].flatten()-1
dataset = data[4].flatten()
n = len(ids)

print(labels)

# Load images
x_train = np.stack([images[:,:,i,None] for i in range(n) if dataset[i]==TRAIN])
x_val = np.stack([images[:,:,i,None] for i in range(n) if dataset[i]==VAL])
x_test = np.stack([images[:,:,i,None] for i in range(n) if dataset[i]==TEST])

# Load labels
y_train = np.stack([labels[i] for i in range(n) if dataset[i]==TRAIN])
y_val = np.stack([labels[i] for i in range(n) if dataset[i]==VAL])
y_test = np.stack([labels[i] for i in range(n) if dataset[i]==TEST])

print("Loaded {0} training samples".format(len(y_train)))
print("Loaded {0} validation samples".format(len(y_val)))
print("Loaded {0} testing samples".format(len(y_test)))

# Generate dummy data
#x_train = np.random.random((100, 100, 100, 3))
#y_train = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)
#x_test = np.random.random((20, 100, 100, 3))
#y_test = keras.utils.to_categorical(np.random.randint(10, size=(20, 1)), num_classes=10)




from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping

import numpy as np
import resnet

batch_size = 128
nb_classes = 2
nb_epoch = 200
data_augmentation = False

# input image dimensions
img_rows, img_cols = 32, 32

# The GDXRay images are Grayscale.
img_channels = 1

# Convert class vectors to binary class matrices.
y_train_c = np_utils.to_categorical(y_train, num_classes=nb_classes)
y_val_c = np_utils.to_categorical(y_val, num_classes=nb_classes)
y_test_c = np_utils.to_categorical(y_test, num_classes=nb_classes)

model = resnet.ResnetBuilder.build_resnet_18((img_channels, img_rows, img_cols), nb_classes)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


while True:
    print('Not using data augmentation.')
    model.fit(x_train, y_train_c,
              batch_size=batch_size,
              epochs=1,
              validation_data=(x_test, y_test_c),
              shuffle=True
    )

    y_pred = model.predict(x_test, batch_size=32)
    print(y_pred.shape)
    y_pred = np.argmax(y_pred, axis=1)
    print(y_pred.shape)
    print(y_pred)
    print(confusion_matrix(y_test, y_pred))

