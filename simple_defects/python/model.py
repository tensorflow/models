from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

def build_model(input_shape,flatten):
	model = Sequential()

	# input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
	# this applies 32 convolution filters of size 3x3 each.
	model.add(Conv2D(32, (7, 7), input_shape=input_shape))
	model.add(Conv2D(32, (7, 7), input_shape=input_shape))
	#model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Conv2D(64, (5, 5), padding='same'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(64, (3, 3)))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(128, (2, 2), activation='relu'))
	model.add(Conv2D(2, (1, 1), activation='softmax'))

	if flatten:
		model.add(Flatten())

	outputs = [layer.output for layer in model.layers]
	print("Input layer", model.layers[0].input)
	print("Output layer", model.layers[-1].output)
	print(model.summary())
	return model