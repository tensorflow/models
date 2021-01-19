import tensorflow_datasets as tfds
from matplotlib import patches
from matplotlib import pyplot as plt
from official.vision.beta.projects.yolo.dataloaders.decoders import \
  tfds_coco_decoder
from official.vision.beta.projects.yolo.utils import box_utils

from official.vision.beta.projects.yolo.dataloaders import yolo_detection_input

dataset = tfds.load("coco", split="train")

decoder = tfds_coco_decoder.MSCOCODecoder()
parser = yolo_detection_input.Parser()

dataset = dataset.map(decoder.decode)
imgs = dataset.map(parser._parse_train_data)

for img, label in imgs.take(10):
  fig, ax = plt.subplots(1)
  # Display the image
  ax.imshow(img)
  # Create a Rectangle patch
  for bbox in label["bbox"].numpy():
    bbox = box_utils.xcycwh_to_xyxy(bbox).numpy()
    wh = bbox[2:4] - bbox[0:2]
    rect = patches.Rectangle(bbox[0:2] * 416,
                             wh[0] * 416,
                             wh[1] * 416,
                             linewidth=1,
                             edgecolor="r",
                             facecolor="none")
    ax.add_patch(rect)
    # Add the patch to the Axes
  plt.show()
