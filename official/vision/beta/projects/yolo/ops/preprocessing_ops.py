import tensorflow as tf
import tensorflow_addons as tfa
from official.vision.beta.projects.yolo.ops import box_ops


def resize_crop_filter(image, boxes, default_width, default_height,
                       target_width, target_height):
  """Apply zooming to the image and boxes.
  Args:
      image: a `Tensor` representing the image.
      boxes: a `Tensor` represeting the boxes.
      default_width: a `Tensor` representing the width of the image.
      default_height: a `Tensor` representing the height of the image.
      target_width: a `Tensor` representing the desired width of the image.
      target_height: a `Tensor` representing the desired height of the image.
  Returns:
      images: a `Tensor` representing the augmented image.
      boxes: a `Tensor` representing the augmented boxes.
  """
  with tf.name_scope('resize_crop_filter'):
    image = tf.image.resize(image, (target_width, target_height))
    image = tf.image.resize_with_crop_or_pad(
        image, target_height=default_height, target_width=default_width)

    default_width = tf.cast(default_width, boxes.dtype)
    default_height = tf.cast(default_height, boxes.dtype)
    target_width = tf.cast(target_width, boxes.dtype)
    target_height = tf.cast(target_height, boxes.dtype)

    aspect_change_width = target_width / default_width
    aspect_change_height = target_height / default_height

    x, y, width, height = tf.split(boxes, 4, axis=-1)
    x = (x - 0.5) * target_width / default_width + 0.5
    y = (y - 0.5) * target_height / default_height + 0.5
    width = width * aspect_change_width
    height = height * aspect_change_height
    boxes = tf.concat([x, y, width, height], axis=-1)
  return image, boxes


def random_translate(image, box, t, seed=None):
  """Randomly translate the image and boxes.
  Args:
      image: a `Tensor` representing the image.
      box: a `Tensor` represeting the boxes.
      t: an `int` representing the translation factor
      seed: an optional seed for tf.random operations
  Returns:
      image: a `Tensor` representing the augmented image.
      box: a `Tensor` representing the augmented boxes.
  """
  t_x = tf.random.uniform(
      minval=-t, maxval=t, shape=(), dtype=tf.float32, seed=seed)
  t_y = tf.random.uniform(
      minval=-t, maxval=t, shape=(), dtype=tf.float32, seed=seed)
  box = translate_boxes(box, t_x, t_y)
  image = translate_image(image, t_x, t_y)
  return image, box


def translate_boxes(box, translate_x, translate_y):
  """Randomly translate the boxes.
  Args:
      boxes: a `Tensor` represeitng the boxes.
      translate_x: a `Tensor` represting the translation on the x-axis.
      translate_y: a `Tensor` represting the translation on the y-axis.
  Returns:
      box: a `Tensor` representing the augmented boxes.
  """
  with tf.name_scope('translate_boxs'):
    x = box[..., 0] + translate_x
    y = box[..., 1] + translate_y
    box = tf.stack([x, y, box[..., 2], box[..., 3]], axis=-1)
    box.set_shape([None, 4])
  return box


def translate_image(image, translate_x, translate_y):
  """Randomly translate the image.
  Args:
      image: a `Tensor` representing the image.
      translate_x: a `Tensor` represting the translation on the x-axis.
      translate_y: a `Tensor` represting the translation on the y-axis.
  Returns:
      box: a `Tensor` representing the augmented boxes.
  """
  with tf.name_scope('translate_image'):
    if (translate_x != 0 and translate_y != 0):
      image_jitter = tf.convert_to_tensor([translate_x, translate_y])
      image_jitter.set_shape([2])
      image = tfa.image.translate(
          image, image_jitter * tf.cast(tf.shape(image)[1], tf.float32))
  return image


def pad_max_instances(value, instances, pad_value=0, pad_axis=0):
  shape = tf.shape(value)
  dim1 = shape[pad_axis]
  take = tf.math.reduce_min([instances, dim1])
  value, _ = tf.split(
      value, [take, -1], axis=pad_axis)  # value[:instances, ...]
  pad = tf.convert_to_tensor([tf.math.reduce_max([instances - dim1, 0])])
  nshape = tf.concat([shape[:pad_axis], pad, shape[(pad_axis + 1):]], axis=0)
  pad_tensor = tf.fill(nshape, tf.cast(pad_value, dtype=value.dtype))
  value = tf.concat([value, pad_tensor], axis=pad_axis)
  return value


def fit_preserve_aspect_ratio(image,
                              boxes,
                              width=None,
                              height=None,
                              target_dim=None):
  """Resizes the image while peserving the image aspect ratio.
  Args:
      image: a `Tensor` representing the image.
      box: a `Tensor` representing the boxes.
  Returns:
      image: a `Tensor` representing the image.
      box: a `Tensor` representing the boxes.
  """
  if width is None or height is None:
    shape = tf.shape(image)
    if tf.shape(shape)[0] == 4:
      width = shape[1]
      height = shape[2]
    else:
      width = shape[0]
      height = shape[1]

  clipper = tf.math.maximum(width, height)
  if target_dim is None:
    target_dim = clipper

  pad_width = clipper - width
  pad_height = clipper - height
  image = tf.image.pad_to_bounding_box(image, pad_width // 2, pad_height // 2,
                                       clipper, clipper)

  boxes = box_ops.yxyx_to_xcycwh(boxes)
  x, y, w, h = tf.split(boxes, 4, axis=-1)

  y *= tf.cast(width / clipper, tf.float32)
  x *= tf.cast(height / clipper, tf.float32)

  y += tf.cast((pad_width / clipper) / 2, tf.float32)
  x += tf.cast((pad_height / clipper) / 2, tf.float32)

  h *= tf.cast(width / clipper, tf.float32)
  w *= tf.cast(height / clipper, tf.float32)

  boxes = tf.concat([x, y, w, h], axis=-1)

  boxes = box_ops.xcycwh_to_yxyx(boxes)
  image = tf.image.resize(image, (target_dim, target_dim))
  return image, boxes


def get_best_anchor(y_true, anchors, width=1, height=1):
  """Gets the correct anchor that is assoiciated with each box using IOU between
     input anchors and ground truth.
  Args:
      y_true: tf.Tensor[] for the list of bounding boxes in the yolo format
      anchors: list or tensor for the anchor boxes to be used in prediction
          found via Kmeans
      size: size of the image that the bounding boxes were selected at 416 is
          the default for the original YOLO model
  return:
      tf.Tensor: y_true with the anchor associated with each ground truth box
          known
  """
  with tf.name_scope('get_anchor'):
    width = tf.cast(width, dtype=tf.float32)
    height = tf.cast(height, dtype=tf.float32)

    anchor_xy = y_true[..., 0:2]
    true_wh = y_true[..., 2:4]

    # scale thhe boxes
    anchors = tf.convert_to_tensor(anchors, dtype=tf.float32)
    anchors_x = anchors[..., 0] / width
    anchors_y = anchors[..., 1] / height
    anchors = tf.stack([anchors_x, anchors_y], axis=-1)

    # build a matrix of anchor boxes
    anchors = tf.transpose(anchors, perm=[1, 0])
    anchor_xy = tf.tile(
        tf.expand_dims(anchor_xy, axis=-1), [1, 1, tf.shape(anchors)[-1]])
    anchors = tf.tile(
        tf.expand_dims(anchors, axis=0), [tf.shape(anchor_xy)[0], 1, 1])

    # stack the xy so, each anchor is asscoaited once with each center from
    # the ground truth input
    anchors = tf.keras.layers.concatenate([anchor_xy, anchors], axis=1)
    anchors = tf.transpose(anchors, perm=[2, 0, 1])

    # copy the gt n times so that each anchor from above can be compared to
    # input ground truth
    truth_comp = tf.tile(
        tf.expand_dims(y_true[..., 0:4], axis=-1),
        [1, 1, tf.shape(anchors)[0]])
    truth_comp = tf.transpose(truth_comp, perm=[2, 0, 1])

    # compute intersection over union of the boxes, and take the argmax of
    # comuted iou for each box. thus each box is associated with the largest
    # interection over union
    iou_raw = box_ops.compute_iou(truth_comp, anchors)

    gt_mask = tf.cast(iou_raw > 0.213, dtype=iou_raw.dtype)

    num_k = tf.reduce_max(
        tf.reduce_sum(tf.transpose(gt_mask, perm=[1, 0]), axis=1))
    if num_k <= 0:
      num_k = 1.0

    values, indexes = tf.math.top_k(
        tf.transpose(iou_raw, perm=[1, 0]),
        k=tf.cast(num_k, dtype=tf.int32),
        sorted=True)
    ind_mask = tf.cast(values > 0.213, dtype=indexes.dtype)
    iou_index = tf.concat([
        tf.expand_dims(indexes[..., 0], axis=-1),
        ((indexes[..., 1:] + 1) * ind_mask[..., 1:]) - 1
    ],
                          axis=-1)

    stack = tf.zeros(
        [tf.shape(iou_index)[0],
         tf.cast(1, dtype=iou_index.dtype)],
        dtype=iou_index.dtype) - 1
    while num_k < 5:
      iou_index = tf.concat([iou_index, stack], axis=-1)
      num_k += 1
    iou_index = iou_index[..., :5]

    values = tf.concat([
        tf.expand_dims(values[..., 0], axis=-1),
        ((values[..., 1:]) * tf.cast(ind_mask[..., 1:], dtype=tf.float32))
    ],
                       axis=-1)
  return tf.cast(iou_index, dtype=tf.float32)


def build_grided_gt(y_true, mask, size, num_classes, dtype, use_tie_breaker):
  """convert ground truth for use in loss functions
  Args:
      y_true: tf.Tensor[] ground truth
        [box coords[0:4], classes_onehot[0:-1], best_fit_anchor_box]
      mask: list of the anchor boxes choresponding to the output,
        ex. [1, 2, 3] tells this layer to predict only the first 3 anchors
        in the total.
      size: the dimensions of this output, for regular, it progresses from
        13, to 26, to 52

  Return:
      tf.Tensor[] of shape [size, size, #of_anchors, 4, 1, num_classes]
  """
  boxes = tf.cast(y_true['bbox'], dtype)
  classes = tf.one_hot(
      tf.cast(y_true['classes'], dtype=tf.int32),
      depth=num_classes,
      dtype=dtype)
  anchors = tf.cast(y_true['best_anchors'], dtype)

  num_boxes = tf.shape(boxes)[0]
  len_masks = tf.shape(mask)[0]

  full = tf.zeros([size, size, len_masks, num_classes + 4 + 1], dtype=dtype)
  depth_track = tf.zeros((size, size, len_masks), dtype=tf.int32)

  x = tf.cast(boxes[..., 0] * tf.cast(size, dtype=dtype), dtype=tf.int32)
  y = tf.cast(boxes[..., 1] * tf.cast(size, dtype=dtype), dtype=tf.int32)

  anchors = tf.repeat(tf.expand_dims(anchors, axis=-1), len_masks, axis=-1)

  update_index = tf.TensorArray(tf.int32, size=0, dynamic_size=True)
  update = tf.TensorArray(dtype, size=0, dynamic_size=True)
  const = tf.cast(tf.convert_to_tensor([1.]), dtype=dtype)
  mask = tf.cast(mask, dtype=dtype)

  i = 0
  anchor_id = 0
  for box_id in range(num_boxes):
    if tf.keras.backend.all(tf.math.equal(boxes[box_id, 2:4], 0)):
      continue
    if tf.keras.backend.any(tf.math.less(
        boxes[box_id, 0:2], 0.0)) or tf.keras.backend.any(
            tf.math.greater_equal(boxes[box_id, 0:2], 1.0)):
      continue
    if use_tie_breaker:
      for anchor_id in range(tf.shape(anchors)[-1]):
        index = tf.math.equal(anchors[box_id, anchor_id], mask)
        if tf.keras.backend.any(index):
          p = tf.cast(
              tf.keras.backend.argmax(tf.cast(index, dtype=tf.int32)),
              dtype=tf.int32)
          uid = 1
          used = depth_track[y[box_id], x[box_id], p]

          if anchor_id == 0:
            # write the box to the update list
            update_index = update_index.write(i, [y[box_id], x[box_id], p])
            value = tf.keras.backend.concatenate(
                [boxes[box_id], const, classes[box_id]])
            update = update.write(i, value)
          elif tf.math.equal(used, 2) or tf.math.equal(used, 0):
            # write the box to the update list
            uid = 2
            update_index = update_index.write(i, [y[box_id], x[box_id], p])
            value = tf.keras.backend.concatenate(
                [boxes[box_id], const, classes[box_id]])
            update = update.write(i, value)

          depth_track = tf.tensor_scatter_nd_update(depth_track,
                                                    [(y[box_id], x[box_id], p)],
                                                    [uid])
          i += 1
    else:
      index = tf.math.equal(anchors[box_id, 0], mask)
      if tf.keras.backend.any(index):
        p = tf.cast(
            tf.keras.backend.argmax(tf.cast(index, dtype=tf.int32)),
            dtype=tf.int32)
        update_index = update_index.write(i, [y[box_id], x[box_id], p])
        value = tf.keras.backend.concatenate(
            [boxes[box_id], const, classes[box_id]])
        update = update.write(i, value)
        i += 1

  # if the size of the update list is not 0, do an update, other wise, no boxes
  # and pass an empty grid
  if tf.math.greater(update_index.size(), 0):
    update_index = update_index.stack()
    update = update.stack()
    full = tf.tensor_scatter_nd_add(full, update_index, update)
  return full


def build_batch_grided_gt(y_true, mask, size, num_classes, dtype,
                          use_tie_breaker):
  """convert ground truth for use in loss functions
  Args:
      y_true: tf.Tensor[] ground truth
          [box coords[0:4], classes_onehot[0:-1], best_fit_anchor_box]
      mask: list of the anchor boxes choresponding to the output,
          ex. [1, 2, 3] tells this layer to predict only the first 3 anchors in
          the total.
      size: the dimensions of this output, for regular, it progresses from
          13, to 26, to 52

  Return:
      tf.Tensor[] of shape [batch, size, size, #of_anchors, 4, 1, num_classes]
    """
  boxes = tf.cast(y_true['bbox'], dtype)
  classes = tf.one_hot(
      tf.cast(y_true['classes'], dtype=tf.int32),
      depth=num_classes,
      dtype=dtype)
  anchors = tf.cast(y_true['best_anchors'], dtype)

  batches = tf.shape(boxes)[0]
  num_boxes = tf.shape(boxes)[1]
  len_masks = tf.shape(mask)[0]

  full = tf.zeros([batches, size, size, len_masks, num_classes + 4 + 1],
                  dtype=dtype)
  depth_track = tf.zeros((batches, size, size, len_masks), dtype=tf.int32)

  x = tf.cast(boxes[..., 0] * tf.cast(size, dtype=dtype), dtype=tf.int32)
  y = tf.cast(boxes[..., 1] * tf.cast(size, dtype=dtype), dtype=tf.int32)

  anchors = tf.repeat(tf.expand_dims(anchors, axis=-1), len_masks, axis=-1)

  update_index = tf.TensorArray(tf.int32, size=0, dynamic_size=True)
  update = tf.TensorArray(dtype, size=0, dynamic_size=True)
  const = tf.cast(tf.convert_to_tensor([1.]), dtype=dtype)
  mask = tf.cast(mask, dtype=dtype)

  i = 0
  anchor_id = 0
  for batch in range(batches):
    for box_id in range(num_boxes):
      if tf.keras.backend.all(tf.math.equal(boxes[batch, box_id, 2:4], 0)):
        continue
      if tf.keras.backend.any(tf.math.less(
          boxes[batch, box_id, 0:2], 0.0)) or tf.keras.backend.any(
              tf.math.greater_equal(boxes[batch, box_id, 0:2], 1.0)):
        continue
      if use_tie_breaker:
        for anchor_id in range(tf.shape(anchors)[-1]):
          index = tf.math.equal(anchors[batch, box_id, anchor_id], mask)
          if tf.keras.backend.any(index):
            #tf.print(anchor_id, anchors[batch, box_id, anchor_id])
            p = tf.cast(
                tf.keras.backend.argmax(tf.cast(index, dtype=tf.int32)),
                dtype=tf.int32)
            uid = 1

            used = depth_track[batch, y[batch, box_id], x[batch, box_id], p]
            if anchor_id == 0:
              # write the box to the update list
              update_index = update_index.write(
                  i, [batch, y[batch, box_id], x[batch, box_id], p])
              value = tf.keras.backend.concatenate(
                  [boxes[batch, box_id], const, classes[batch, box_id]])
              update = update.write(i, value)
            elif tf.math.equal(used, 2) or tf.math.equal(used, 0):
              uid = 2
              # write the box to the update list
              update_index = update_index.write(
                  i, [batch, y[batch, box_id], x[batch, box_id], p])
              value = tf.keras.backend.concatenate(
                  [boxes[batch, box_id], const, classes[batch, box_id]])
              update = update.write(i, value)

            depth_track = tf.tensor_scatter_nd_update(
                depth_track, [(batch, y[batch, box_id], x[batch, box_id], p)],
                [uid])
            i += 1
      else:
        index = tf.math.equal(anchors[batch, box_id, 0], mask)
        if tf.keras.backend.any(index):
          p = tf.cast(
              tf.keras.backend.argmax(tf.cast(index, dtype=tf.int32)),
              dtype=tf.int32)
          update_index = update_index.write(
              i, [batch, y[batch, box_id], x[batch, box_id], p])
          value = tf.keras.backend.concatenate(
              [boxes[batch, box_id], const, classes[batch, box_id]])
          update = update.write(i, value)
          i += 1

  # if the size of the update list is not 0, do an update, other wise, no boxes
  # and pass an empty grid
  if tf.math.greater(update_index.size(), 0):
    update_index = update_index.stack()
    update = update.stack()
    full = tf.tensor_scatter_nd_add(full, update_index, update)
  return full
