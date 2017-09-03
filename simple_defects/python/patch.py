import scipy.io
import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.regularizers import l2
from model import build_model

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



# encode to one hot
y_train_c = to_categorical(y_train, num_classes=2)
y_val_c = to_categorical(y_val, num_classes=2)
y_test_c = to_categorical(y_test, num_classes=2)

model = build_model(INPUT_SHAPE,flatten=True)
optimizer = Adam(lr=0.001, decay=1e-6)
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=optimizer)

for i in range(1000):
	#model.fit(x_train, y_train, batch_size=128, epochs=1)
	model.fit(x_train, y_train_c, batch_size=128, epochs=1)
	y_pred = model.predict(x_test, batch_size=32)
	y_pred = np.argmax(y_pred, axis=1)
	print(confusion_matrix(y_test, y_pred))
	model.save_weights('models/model_%i_weights.h5'%i)


