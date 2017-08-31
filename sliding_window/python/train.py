import scipy.io
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical

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

model = Sequential()

# input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
# this applies 32 convolution filters of size 3x3 each.
model.add(Conv2D(32, (7, 7), input_shape=INPUT_SHAPE))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (5, 5)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Dropout(0.25))

#model.add(Conv2D(64, (2, 2)))
#model.add(Conv2D(2, (1, 1)))
#model.add(Flatten())
#model.add(Dense(2, activation='softmax'))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

# encode to one hot
y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)

optimizer = Adam(lr=0.001, decay=1e-6)
#model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=optimizer)
model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=optimizer)

while True:
	model.fit(x_train, y_train, batch_size=128, epochs=1)
	score = model.evaluate(x_test, y_test, batch_size=32)
	print("score",score)

