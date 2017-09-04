"""
Convolve the trained mode over a larger image
"""
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image
from model import build_model

#INPUT_SHAPE = (1070,942,1)
INPUT_SHAPE = (572,768,1)
MODEL_WEIGHTS = 'models/model_6_weights.h5'
TEST_IMAGES = 'images'

model = build_model(INPUT_SHAPE, flatten=False)
model.load_weights(MODEL_WEIGHTS)
print("Loaded weights from",MODEL_WEIGHTS)


for filename in glob.glob(os.path.join(TEST_IMAGES, '*.png')):
	pixels = matplotlib.image.imread(filename)
	pixels = pixels[None,:,:,None]
	result = model.predict(pixels)
	predict = np.argmax(result, axis=3)
	plt.imshow(predict[0,:,:], cmap='hot', interpolation='nearest')
	plt.show()