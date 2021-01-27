from official.vision.beta.projects.yolo.dataloaders.yolo_detection_input_test import *
from official.vision.beta.projects.yolo.modeling.backbones.darknet_test import *
from official.vision.beta.projects.yolo.modeling.layers.nn_blocks_test import *
from official.vision.beta.projects.yolo.modeling.decoders.yolo_decoder_test import *

# this file can be removed, it just runs all the tests

# def prep_gpu(distribution=None):
#   import tensorflow as tf
#   try:
#     from tensorflow.config import list_physical_devices, list_logical_devices
#   except ImportError:
#     from tensorflow.config.experimental import list_physical_devices, list_logical_devices
#   print(f"\n!--PREPPING GPU--! ")
#   if distribution is None:
#     gpus = list_physical_devices('GPU')
#     if gpus:
#       try:
#         # Currently, memory growth needs to be the same across GPUs
#         for gpu in gpus:
#           tf.config.experimental.set_memory_growth(gpu, True)
#           logical_gpus = list_logical_devices('GPU')
#           print(len(gpus), 'Physical GPUs,', len(logical_gpus), 'Logical GPUs')
#       except RuntimeError as e:
#         # Memory growth must be set before GPUs have been initialized
#         print(e)
#         raise
#   return

if __name__ == "__main__":
  # prep_gpu()
  tf.test.main()

