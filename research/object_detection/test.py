from datasets import dataset_utils
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image
import time

IMAGE_SIZE = (12, 8)
image_path = r"C:\Users\sunhongzhi\Desktop\deep32\1420187_fcfb198d3c8a0bf1fee102cc0c258117.jpg"
with Image.open(image_path) as image:
    (im_width, im_height) = image.size
    if image.mode == "RGBA":
        image = image.convert("RGB")
    print(image.mode)  # RGBA
    image_np = np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)
    plt.figure(figsize=IMAGE_SIZE)
    plt.imshow(image_np)
    plt.show()
    # image_np = load_image_into_numpy_array(image)
