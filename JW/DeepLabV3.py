import os
from io import BytesIO
import tarfile
import tempfile
import numpy as np
from PIL import Image
import cv2

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class DeepLabModel(object):
  """Class to load deeplab model and run inference."""

  INPUT_TENSOR_NAME = 'ImageTensor:0'
  OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
  INPUT_SIZE = 513
  FROZEN_GRAPH_NAME = 'frozen_inference_graph'
  predict_signature_def = None

  def __init__(self, tarball_path):
    """Creates and loads pretrained deeplab model."""
    self.graph = tf.Graph()

    # Extract frozen graph from tar archive.
    tar_file = tarfile.open(tarball_path)
    for tar_info in tar_file.getmembers():

      if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
        file_handle = tar_file.extractfile(tar_info)
        graph_def = tf.GraphDef.FromString(file_handle.read())

        break

    tar_file.close()

    if graph_def is None:
      raise RuntimeError('Cannot find inference graph in tar archive.')

    with self.graph.as_default():
      tf.import_graph_def(graph_def, name='')

    self.sess = tf.Session(graph=self.graph)


  def get_signature_def(self):

    InputTensor = self.graph.get_tensor_by_name(self.INPUT_TENSOR_NAME)
    OutputTensor = self.graph.get_tensor_by_name(self.OUTPUT_TENSOR_NAME)

    predict_input_tensor = tf.saved_model.utils.build_tensor_info(InputTensor)
    predict_signature_inputs = {"x": predict_input_tensor}

    predict_output_tensor = tf.saved_model.utils.build_tensor_info(OutputTensor)
    predict_signature_outputs = {"y": predict_output_tensor}
    predict_signature_def = (
      tf.saved_model.signature_def_utils.build_signature_def(
        predict_signature_inputs, predict_signature_outputs,
        tf.saved_model.signature_constants.PREDICT_METHOD_NAME))
    return predict_signature_def

  def store(self, path):
    #self.sess.run(tf.global_variables_initializer())
    self.sess.run(tf.local_variables_initializer())
    builder = tf.saved_model.builder.SavedModelBuilder(path)
    builder.add_meta_graph_and_variables(
      self.sess, [tf.saved_model.tag_constants.SERVING],
      signature_def_map={
        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
          self.get_signature_def()
      },
      main_op=tf.tables_initializer())
    builder.save()



  def run(self, image):
    """Runs inference on a single image.

    Args:
      image: A PIL.Image object, raw input image.

    Returns:
      resized_image: RGB image resized from original input image.
      seg_map: Segmentation map of `resized_image`.
    """
    width, height = image.size
    resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
    target_size = (int(resize_ratio * width), int(resize_ratio * height))
    resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
    batch_seg_map = self.sess.run(
        self.OUTPUT_TENSOR_NAME,
        feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
    seg_map = batch_seg_map[0]
    return resized_image, seg_map

  def runWithCV(self, image_path):
    image = cv2.imread(image_path)
    height, width, channel = image.shape
    resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
    target_size = (int(resize_ratio * width), int(resize_ratio * height))
    #print(width)
    resized_image = cv2.resize(image, target_size)
    #print(target_size)
    #print(resized_image.shape)
    resized_image = cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR)

    InputTensorNode = self.graph.get_tensor_by_name(self.INPUT_TENSOR_NAME)
    OutputTensorNode = self.graph.get_tensor_by_name(self.OUTPUT_TENSOR_NAME)

    batch_seg_map = self.sess.run(
      OutputTensorNode,
      feed_dict={InputTensorNode: [np.asarray(resized_image)]})
    seg_map = batch_seg_map[0]
    #print(batch_seg_map.size)
    return resized_image, seg_map

'''
#Not Used Function
def LoadGraph(tarball_path):
  """Creates and loads pretrained deeplab model."""
  graph = tf.Graph()
  FROZEN_GRAPH_NAME = 'frozen_inference_graph'

  # Extract frozen graph from tar archive.
  tar_file = tarfile.open(tarball_path)
  for tar_info in tar_file.getmembers():

    if FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
      file_handle = tar_file.extractfile(tar_info)
      graph_def = tf.GraphDef.FromString(file_handle.read())

      #print(graph_def)
      break

  tar_file.close()

  if graph_def is None:
    raise RuntimeError('Cannot find inference graph in tar archive.')

  with graph.as_default():
    tf.import_graph_def(graph_def, name='TempGraph')

  sess = tf.Session(graph=graph)
  return graph, sess
def get_signature_def():
  INPUT_TENSOR_NAME = 'ImageTensor:0'
  OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
  InputTensor = graph.get_tensor_by_name(INPUT_TENSOR_NAME)
  OutputTensor = graph.get_tensor_by_name(OUTPUT_TENSOR_NAME)

  predict_input_tensor = tf.saved_model.utils.build_tensor_info(InputTensor)
  predict_signature_inputs = {"x": predict_input_tensor}

  predict_output_tensor = tf.saved_model.utils.build_tensor_info(OutputTensor)
  predict_signature_outputs = {"y": predict_output_tensor}
  predict_signature_def = (
    tf.saved_model.signature_def_utils.build_signature_def(
      predict_signature_inputs, predict_signature_outputs,
      tf.saved_model.signature_constants.PREDICT_METHOD_NAME))
  return predict_signature_def
def StoreSavedMOdel(sess, path):
  sess.run(tf.global_variables_initializer())
  sess.run(tf.local_variables_initializer())
  builder = tf.saved_model.builder.SavedModelBuilder(path)
  builder.add_meta_graph_and_variables(
    sess, [tf.saved_model.tag_constants.SERVING],
    signature_def_map={
      tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
        get_signature_def()
    },
    main_op=tf.tables_initializer())
  builder.save()
'''

#load frozen_inference_graph(inference model)

#graph2, sess = LoadGraph(download_path1)
#StoreSavedMOdel(sess, saved_model_path)

#print(res)

#graph = tf.get_default_graph()
#print(graph.get_operations())


#MODEL.store(saved_model_path)


#print(MODEL.graph.get_operations())
#print(graph2.get_operations())

#store savedmodel
'''
builder = tf.saved_model.builder.SavedModelBuilder('../model/deeplab_ade20k/1/')
builder.add_meta_graph_and_variables(
      tf.Session(), [tf.saved_model.tag_constants.SERVING],
      None,
      main_op=tf.tables_initializer())
builder.save()
'''

#load savedmodel
'''
sess = tf.Session()
loaded = tf.saved_model.load(sess,[tf.saved_model.tag_constants.SERVING],saved_model_path)
#loaded = tf.saved_model.load_v2(saved_model_path)
graph = tf.get_default_graph()
print(graph.get_operations())
'''
