import tensorflow as tf
import numpy as np
from PIL import Image
import os
import glob

export_dir_path = "/home/gunho1123/export_basnet/saved_model"
#input_img_path = "/home/gunho1123/DUTS-TE/DUTS-TE-Image"
input_img_path = "test.jpg"
#save_dir_path = "/home/gunho1123/DUTS-TE/DUTS-TE-Out"
save_dir_path = "./DUTS-TE-Out"

if not os.path.exists(save_dir_path):
  os.makedirs(save_dir_path)


# Read image file names
full_filenames = []
if os.path.isdir(input_img_path):
  full_filenames = glob.glob(os.path.join(input_img_path, "*.jpg"))
else:
  full_filenames.append(input_img_path)

# Load exported model
imported = tf.saved_model.load(export_dir_path)
model_fn = imported.signatures['serving_default']


for i, name in enumerate(full_filenames):
  input_images = np.array(Image.open(name))
  height = input_images.shape[0]
  width = input_images.shape[1]

  input_images = tf.image.resize(input_images, [256, 256])
  input_images = tf.reshape(input_images, [1, 256, 256, 3])
  processed_images = tf.cast(input_images, tf.uint8)

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
  new_name = os.path.join(save_dir_path, os.path.basename(name).replace(".jpg", ".png"))
  #print(new_name)
  tf.keras.preprocessing.image.save_img(new_name, img, data_format="channels_last", scale=False)
  if i%100 == 0:
    print("progress : "+str(i)+" of "+str(len(full_filenames)))

