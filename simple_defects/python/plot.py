import random
import scipy.io
import keras
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.regularizers import l2

def gallery(array, ncols=3):
    nindex, height, width, intensity = array.shape
    nrows = nindex//ncols
    assert nindex == nrows*ncols
    # want result.shape = (height*nrows, width*ncols, intensity)
    result = (array.reshape((nrows, ncols, height, width, intensity))
              .swapaxes(1,2)
              .reshape((height*nrows, width*ncols, intensity)))
    return result

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
x_train = [images[:,:,i,None] for i in range(n) if dataset[i]==VAL]
y_train = [labels[i] for i in range(n) if dataset[i]==VAL]
print("Loaded {0} training samples".format(len(y_train)))

# Split images by defectness
x_defect = [x_train[i] for i,label in enumerate(y_train) if not label]
x_nodefect = [x_train[i] for i,label in enumerate(y_train) if label]

# Shuffle the elements in the array
random.shuffle(x_defect)
random.shuffle(x_nodefect)

ncols = 6
nrows = 8
x_defect = np.array(x_defect[:ncols*nrows])
x_nodefect = np.array(x_nodefect[:ncols*nrows])

defect = gallery(x_defect, ncols=ncols).squeeze()
ndefect = gallery(x_nodefect, ncols=ncols).squeeze()

plt.imshow(defect**0.5, cmap='gray',vmin=0)
plt.savefig("x_defect.png")

plt.imshow(ndefect**0.4, cmap='gray',vmin=0)
plt.savefig("x_nodefect.png")

#plt.show()
#cv.imwrite("x_defect.png", 255*defect );
#cv.imwrite("x_nodefect.png", 255*ndefect );







