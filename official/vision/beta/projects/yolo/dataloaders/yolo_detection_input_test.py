"""Test case for YOLO detection dataloader configuration definition."""
import dataclasses

import tensorflow as tf
from official.core import config_definitions as cfg
from official.core import input_reader
from official.modeling import hyperparams
from official.vision.beta.projects.yolo.dataloaders import yolo_detection_input
from official.vision.beta.projects.yolo.dataloaders.decoders import \
    tfds_coco_decoder
from official.vision.beta.projects.yolo.ops import box_ops
from absl.testing import parameterized


@dataclasses.dataclass
class Parser(hyperparams.Config):
  """Dummy configuration for parser"""
  output_size: int = (416, 416)
  num_classes: int = 80
  fixed_size: bool = True
  jitter_im: float = 0.1
  jitter_boxes: float = 0.005
  min_process_size: int = 320
  max_process_size: int = 608
  max_num_instances: int = 200
  random_flip: bool = True
  seed: int = 10
  shuffle_buffer_size: int = 10000


@dataclasses.dataclass
class DataConfig(cfg.DataConfig):
  """Input config for training."""
  input_path: str = ''
  tfds_name: str = 'coco/2017'
  tfds_split: str = 'train'
  global_batch_size: int = 10
  is_training: bool = True
  dtype: str = 'float16'
  decoder = None
  parser: Parser = Parser()
  shuffle_buffer_size: int = 10000
  tfds_download: bool = True


class YoloDetectionInputTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(('training', True), ('testing', False))
  def test_yolo_input(self, is_training):
    with tf.device('/CPU:0'):
      params = DataConfig(is_training=is_training)

      decoder = tfds_coco_decoder.MSCOCODecoder()
      anchors = [[12.0, 19.0], [31.0, 46.0], [96.0, 54.0], [46.0, 114.0],
                 [133.0, 127.0], [79.0, 225.0], [301.0, 150.0], [172.0, 286.0],
                 [348.0, 340.0]]
      masks = {'3': [0, 1, 2], '4': [3, 4, 5], '5': [6, 7, 8]}

      parser = yolo_detection_input.Parser(
          output_size=params.parser.output_size,
          num_classes=params.parser.num_classes,
          fixed_size=params.parser.fixed_size,
          jitter_im=params.parser.jitter_im,
          jitter_boxes=params.parser.jitter_boxes,
          min_process_size=params.parser.min_process_size,
          max_process_size=params.parser.max_process_size,
          max_num_instances=params.parser.max_num_instances,
          random_flip=params.parser.random_flip,
          seed=params.parser.seed,
          anchors=anchors,
          masks=masks)
      postprocess_fn = parser.postprocess_fn(is_training=is_training)

      reader = input_reader.InputReader(
          params,
          dataset_fn=tf.data.TFRecordDataset,
          decoder_fn=decoder.decode,
          parser_fn=parser.parse_fn(params.is_training))
      dataset = reader.read(input_context=None)
      for one_batch in dataset.batch(1):
        self.assertAllEqual(one_batch[0].shape, (1, 10, 416, 416, 3))
        break

      for l, (i, j) in enumerate(dataset):
        if postprocess_fn:
          i, j = postprocess_fn(i, j)
        boxes = box_ops.xcycwh_to_yxyx(j['bbox'])
        self.assertTrue(tf.reduce_all(tf.math.logical_and(i >= 0, i <= 1)))
        if l > 10:
          break


if __name__ == '__main__':
  tf.test.main()
