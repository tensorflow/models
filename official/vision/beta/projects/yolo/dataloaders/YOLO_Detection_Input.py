""" Detection Data parser and processing for YOLO.
Parse image and ground truths in a dataset to training targets and package them
into (image, labels) tuple for RetinaNet.
"""

# Import libraries
import tensorflow as tf
from official.vision.beta.dataloaders import parser
from official.vision.beta.ops import box_ops, preprocess_ops
from official.vision.beta.ops import preprocessing_ops
from official.vision.beta.projects.yolo.utils import box_ops as box_utils


class Parser(parser.Parser):
  """Parser to parse an image and its annotations into a dictionary of tensors."""
  def __init__(
      self,
      image_w=416,
      image_h=416,
      num_classes=80,
      fixed_size=False,
      jitter_im=0.1,
      jitter_boxes=0.005,
      net_down_scale=32,
      max_process_size=608,
      min_process_size=320,
      max_num_instances=200,
      random_flip=True,
      pct_rand=0.5,
      anchors=None,
      seed=10,
  ):
    """Initializes parameters for parsing annotations in the dataset.
        Args:
            image_w: a `Tensor` or `int` for width of input image.
            image_h: a `Tensor` or `int` for height of input image.
            num_classes: a `Tensor` or `int` for the number of classes.
            fixed_size: a `bool` if True all output images have the same size.
            jitter_im: a `float` that is the maximum jitter applied to the image
                for data augmentation during training.
            jitter_boxes: a `float` that is the maximum jitter applied to the
                bounding box for data augmentation during training.
            net_down_scale: an `int` that down scales the image width and height
                to the closest multiple of net_down_scale.
            max_process_size: an `int` for maximum image width and height.
            min_process_size: an `int` for minimum image width and height ,
            max_num_instances: an `int` number of maximum number of instances in
                an image.
            random_flip: a `bool` if True, augment training with random
                horizontal flip.
            pct_rand: an `int` that prevents do_scale from becoming larger than
                1-pct_rand.
            masks: a `Tensor`, `List` or `numpy.ndarrray` for anchor masks.
            anchors: a `Tensor`, `List` or `numpy.ndarrray` for bounding box
                priors.
            seed: an `int` for the seed used by tf.random
        """
    self._net_down_scale = net_down_scale
    self._image_w = (image_w // self._net_down_scale) * self._net_down_scale
    self._image_h = self.image_w if image_h == None else (
        image_h // self._net_down_scale) * self._net_down_scale
    self._max_process_size = max_process_size
    self._min_process_size = min_process_size
    self._anchors = anchors

    self._fixed_size = fixed_size
    self._jitter_im = 0.0 if jitter_im == None else jitter_im
    self._jitter_boxes = 0.0 if jitter_boxes == None else jitter_boxes
    self._pct_rand = pct_rand
    self._max_num_instances = max_num_instances
    self._random_flip = random_flip
    self._seed = seed

  def _parse_train_data(self, data):
    """Generates images and labels that are usable for model training.
        Args:
            data: a dict of Tensors produced by the decoder.
        Returns:
            images: the image tensor.
            labels: a dict of Tensors that contains labels.
        """

    shape = tf.shape(data["image"])
    image = data["image"] / 255
    image = tf.image.resize(image,
                            (self._max_process_size, self._max_process_size))
    image = tf.image.random_brightness(image=image, max_delta=.1)  # Brightness
    image = tf.image.random_saturation(image=image, lower=0.75,
                                       upper=1.25)  # Saturation
    image = tf.image.random_hue(image=image, max_delta=.1)  # Hue
    image = tf.clip_by_value(image, 0.0, 1.0)
    boxes = data["groundtruth_boxes"]
    image_shape = tf.shape(image)[:2]

    self._random_flip = False
    if self._random_flip:
      image, boxes, _ = preprocess_ops.random_horizontal_flip(image,
                                                              boxes,
                                                              seed=self._seed)

    randscale = self._image_w // self._net_down_scale

    if not self._fixed_size:
      do_scale = tf.greater(
          tf.random.uniform([], minval=0, maxval=1, seed=self._seed),
          1 - self._pct_rand)
      if do_scale:
        randscale = tf.random.uniform([],
                                      minval=10,
                                      maxval=21,
                                      seed=self._seed,
                                      dtype=tf.int32)

    if self._jitter_boxes != 0.0:
      boxes = box_ops.denormalize_boxes(boxes, image_shape)
      boxes = box_ops.jitter_boxes(boxes, 0.025)
      boxes = box_ops.normalize_boxes(boxes, image_shape)

    boxes = box_utils.yxyx_to_xcycwh(boxes)

    if self._jitter_im != 0.0:
      image, boxes = preprocessing_ops.random_translate(image,
                                                        boxes,
                                                        self._jitter_im,
                                                        seed=self._seed)

    image, boxes = preprocessing_ops.resize_crop_filter(
        image,
        boxes,
        default_width=self._image_w,
        default_height=self._image_h,
        target_width=randscale * self._net_down_scale,
        target_height=randscale * self._net_down_scale)

    best_anchors = preprocessing_ops.get_best_anchor(boxes,
                                                     self._anchors,
                                                     width=self._image_w,
                                                     height=self._image_h)

    #padding
    boxes = preprocess_ops.clip_or_pad_to_fixed_size(boxes,
                                                     self._max_num_instances,
                                                     0)
    classes = preprocess_ops.clip_or_pad_to_fixed_size(
        data["groundtruth_classes"], self._max_num_instances, -1)
    best_anchors = preprocess_ops.clip_or_pad_to_fixed_size(
        best_anchors, self._max_num_instances, 0)
    area = preprocess_ops.clip_or_pad_to_fixed_size(data["groundtruth_area"],
                                                    self._max_num_instances, 0)
    is_crowd = preprocess_ops.clip_or_pad_to_fixed_size(
        tf.cast(data["groundtruth_is_crowd"], tf.int32),
        self._max_num_instances, 0)
    labels = {
        "source_id": data["source_id"],
        "bbox": boxes,
        "classes": classes,
        "area": area,
        "is_crowd": is_crowd,
        "best_anchors": best_anchors,
        "width": shape[1],
        "height": shape[2],
        "num_detections": tf.shape(data["groundtruth_classes"])[0],
    }
    return image, labels

  def _parse_eval_data(self, data):
    """Generates images and labels that are usable for model training.
        Args:
            data: a dict of Tensors produced by the decoder.
        Returns:
            images: the image tensor.
            labels: a dict of Tensors that contains labels.
        """

    shape = tf.shape(data["image"])
    image = tf.convert_to_tensor(data["image"])
    image = tf.image.resize(image, size=(self._image_w, self._image_h))
    image = image / 255
    boxes = box_utils.yxyx_to_xcycwh(data["groundtruth_boxes"])
    best_anchors = preprocessing_ops.get_best_anchor(boxes,
                                                     self._anchors,
                                                     width=self._image_w,
                                                     height=self._image_h)
    #padding
    boxes = preprocess_ops.clip_or_pad_to_fixed_size(boxes,
                                                     self._max_num_instances,
                                                     0)
    classes = preprocess_ops.clip_or_pad_to_fixed_size(
        data["groundtruth_classes"], self._max_num_instances, -1)
    best_anchors = preprocess_ops.clip_or_pad_to_fixed_size(
        best_anchors, self._max_num_instances, 0)
    area = preprocess_ops.clip_or_pad_to_fixed_size(data["groundtruth_area"],
                                                    self._max_num_instances, 0)
    is_crowd = preprocess_ops.clip_or_pad_to_fixed_size(
        tf.cast(data["groundtruth_is_crowd"], tf.int32),
        self._max_num_instances, 0)
    labels = {
        "source_id": data["source_id"],
        "bbox": boxes,
        "classes": classes,
        "area": area,
        "is_crowd": is_crowd,
        "best_anchors": best_anchors,
        "width": shape[1],
        "height": shape[2],
        "num_detections": tf.shape(data["groundtruth_classes"])[0],
    }
    return image, labels
