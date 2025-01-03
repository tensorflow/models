import cv2
from PIL import Image
import os

in_files = '/Users/sca/r2k9/dogs'
out_files = '/Users/sca/r2k9/scaled_dogs'

# 6000 4000
no_img = 0
size = 1200, 800
no_skip = 0
for f in os.listdir(in_files):
    if f.lower().endswith('.jpg'):
        img = Image.open(in_files + '/' + f)
        outfile = out_files + '/' + f
        if not os.path.isfile(outfile):
            print('processing {}'.format(outfile))
            img.thumbnail(size)
            img.save(outfile, "JPEG")
            no_img += 1
        else:
            #print('image {} skipped'.format(outfile))
            no_skip += 1

print('processed {} images and skipped {}'.format(no_img, no_skip))
