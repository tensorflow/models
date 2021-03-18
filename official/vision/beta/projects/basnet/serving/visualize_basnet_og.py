import tensorflow as tf
import numpy as np
from PIL import Image



export_dir_path = "/home/gunho1123/export_basnet/saved_model"
input_images = np.array(Image.open('test.jpg'))

height = input_images.shape[0]
width = input_images.shape[1]

input_images = tf.image.resize(input_images, [256, 256])
input_images = tf.reshape(input_images, [1, 256, 256, 3])
processed_images = tf.cast(input_images, tf.uint8)

imported = tf.saved_model.load(export_dir_path)



model_fn = imported.signatures['serving_default']
output = model_fn(processed_images)


output = output['outputs']
output = output*255

output = tf.cast(output, tf.uint8)

output = output[0,:,:,0].numpy()


img = np.zeros((256,256,3))
img[:,:,0] = output
img[:,:,1] = output
img[:,:,2] = output

img = tf.image.resize(img, [height, width])


tf.keras.preprocessing.image.save_img("./output.png", img, data_format="channels_last", scale=False)
