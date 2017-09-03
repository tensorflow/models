import random
import scipy.io
import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.regularizers import l2

TRAIN = 1
TEST = 3
VAL = 2

NUM_CLASSES = 2
INPUT_SHAPE = (32,32,1)

# Load data
data = scipy.io.loadmat("data/imdb.mat")
data = data['imdb'][0][0][0][0][0]
ids = data[0].flatten()
images = data[1]
labels = data[2].flatten()-1
dataset = data[4].flatten()
n = len(ids)

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


"""
for i in range(10):
	print(x_train[i+1000,:,:,0])
	plt.imshow(x_train[i+1000,:,:,0], cmap='gray')
	plt.show()
"""


# Generate dummy data
#x_train = np.random.random((100, 100, 100, 3))
#y_train = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)
#x_test = np.random.random((20, 100, 100, 3))
#y_test = keras.utils.to_categorical(np.random.randint(10, size=(20, 1)), num_classes=10)

model = Sequential()

# input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
# this applies 32 convolution filters of size 3x3 each.
model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=INPUT_SHAPE))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(NUM_CLASSES, W_regularizer=l2(0.01)))
model.add(Dense(2, activation='softmax'))

# encode to one hot
y_train_c = to_categorical(y_train, num_classes=2)
y_val_c = to_categorical(y_val, num_classes=2)
y_test_c = to_categorical(y_test, num_classes=2)

optimizer = Adam(lr=0.001, decay=1e-6)
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=optimizer)

while True:
	#model.fit(x_train, y_train, batch_size=128, epochs=1)
	x_train_noise = x_test + np.random.normal(loc=0.0, scale=0.001, size=x_test.shape)
	model.fit(x_train_noise, y_test_c, batch_size=128, epochs=1)
	y_pred = model.predict(x_test, batch_size=32)
	y_pred = np.argmax(y_pred, axis=1)
	#print(confusion_matrix(y_test, y_pred))

	# Display one of the incorrect items
	errors = y_pred != y_test
	indices = np.arange(len(errors))
	fn = np.all((errors, y_test), axis=0).astype(np.int8)
	fp = np.all((errors, 1-y_test), axis=0).astype(np.int8)
	print(fn)

	fp = indices[fn]
	fn = indices[fp]

	fp = random.choice(fp)
	fn = random.choice(fn)

	fp = x_train_noise[:,:,fp,0]
	fn = x_train_noise[:,:,fn,0]

	plt.imshow(fp, cmap='gray')
	plt.title('False positive')
	plt.show()

	plt.imshow(fn, cmap='gray')
	plt.title('False negative')
	plt.show()






