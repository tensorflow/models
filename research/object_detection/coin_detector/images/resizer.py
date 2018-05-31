## Bulk image resizer

# This script simply resizes all the images in a folder to one-eigth their
# original size. It's useful for shrinking large cell phone pictures down
# to a size that's more manageable for model training.

# Usage: place this script in a folder of images you want to shrink,
# and then run it.

import numpy as np
import cv2
import os

dir_path = os.getcwd()

for filename in os.listdir(dir_path):
    # If the images are not .JPG images, change the line below to match the image type.
    if filename.endswith(".jpg"):

        image = cv2.imread(filename)
        
        print(image.shape)
        
        if image.shape[1] > 1024:
            r = 1024 / image.shape[1]
            dim = (1024, int(image.shape[0] * r))
            print('resizing ' + filename)
            resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)    
            cv2.imwrite(filename, resized)
        elif image.shape[0] > 1024:
            r = 1024 / image.shape[0]
            dim = (int(image.shape[1] * r), 1024)
            print('resizing ' + filename)
            resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
            cv2.imwrite(filename, resized)
        else:
            print('not resizing ' + filename)

