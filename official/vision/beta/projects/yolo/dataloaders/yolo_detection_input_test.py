import dataclasses

import tensorflow as tf
from official.core import config_definitions as cfg
from official.core import input_reader
from official.modeling import hyperparams
from official.vision.beta.projects.yolo.dataloaders import yolo_detection_input
from official.vision.beta.projects.yolo.dataloaders.decoders import \
    tfds_coco_decoder
from official.vision.beta.projects.yolo.utils import box_ops


@dataclasses.dataclass
class Parser(hyperparams.Config):
  image_w: int = 416
  fixed_size: bool = True
  jitter_im: float = 0.1
  jitter_boxes: float = 0.005
  net_down_scale: int = 32
  min_process_size: int = 320
  max_process_size: int = 608
  max_num_instances: int = 200
  random_flip: bool = True
  pct_rand: float = 1.0
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


class yoloDetectionInputTest(tf.test.TestCase):

  def test_yolo_input(self):
    with tf.device('/CPU:0'):
      params = DataConfig(is_training=True)
      num_boxes = 9

      decoder = tfds_coco_decoder.MSCOCODecoder()
      anchors = [[12.0, 19.0], [31.0, 46.0], [96.0, 54.0], [46.0, 114.0],
                 [133.0, 127.0], [79.0, 225.0], [301.0, 150.0], [172.0, 286.0],
                 [348.0, 340.0]]
      masks = {'3': {0, 1, 2}, '4': {3, 4, 5}, '5': {6, 7, 8}}

      parser = yolo_detection_input.Parser(
          image_w=params.parser.image_w,
          fixed_size=params.parser.fixed_size,
          jitter_im=params.parser.jitter_im,
          jitter_boxes=params.parser.jitter_boxes,
          net_down_scale=params.parser.net_down_scale,
          min_process_size=params.parser.min_process_size,
          max_process_size=params.parser.max_process_size,
          max_num_instances=params.parser.max_num_instances,
          random_flip=params.parser.random_flip,
          pct_rand=params.parser.pct_rand,
          seed=params.parser.seed,
          anchors=anchors,
          masks=masks)

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
        boxes = box_ops.xcycwh_to_yxyx(j['bbox'])
        self.assertTrue(tf.reduce_all(tf.math.logical_and(i >= 0, i <= 1)))
        if l > 10:
          break


if __name__ == '__main__':
  tf.test.main()
