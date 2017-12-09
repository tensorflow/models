"""YOLO Meta-architecture definition.

General tensorflow implementation of YOLO models
"""
from abc import abstractmethod

import re
import tensorflow as tf

from object_detection.core import box_list
from object_detection.core import box_list_ops
from object_detection.core import model
from object_detection.core import standard_fields as fields
from object_detection.utils import shape_utils
import tensorflow.contrib.slim as slim

#slim = tf.contrib.slim

class YOLOFeatureExtractor(object):
  """YOLO Feature Extractor definition"""

  def __init__(self,
               is_training,
               reuse_weights,):
    # TODO : find out the parameters
    self.is_training = is_training
    self.reuse_weights = reuse_weights
    pass
  
  @abstractmethod
  def preprocess(self, resized_inputs):
    """Preprocesses images for feature extraction (minus image resizing).

    Args:
      resized_inputs: a [batch, height, width, channels] float tensor
        representing a batch of images.

    Returns:
      preprocessed_inputs: a [batch, height, width, channels] float tensor
        representing a batch of images.
    """
    pass
  
  @abstractmethod
  def extract_features(self, preprocessed_inputs):
    """Extracts features from preprocessed inputs.

    This function is responsible for extracting the YOLO feature map from preprocessed
    images.

    Args:
      preprocessed_inputs: a [batch, height, width, channels] float tensor
        representing a batch of images.

    Returns:
      feature_maps: a tensor where the tensor has shape
        [batch, grid_size, grid_size, depth]
    """
    pass
  

class YOLOMetaArch(model.DetectionModel):
  """YOLO Meta Arch definition"""

  def __init__(self,
              is_training,
              feature_extractor,
              matcher,
              num_classes,
              region_similarity_calculator,
              image_resizer_fn,
              non_max_suppression_fn,
              score_conversion_fn,
              localization_loss_weight,
              noobject_loss_weight,
              add_summaries=True):
    """YOLOMetaArch Constructor

    Args:
      is_training: A boolean indicating whether the training version of the
        computation graph should be constructed.
      feature_extractor: a YOLOFeatureExtractor object.
      matcher: a matcher.Matcher object.
      region_similarity_calculator: a
        region_similarity_calculator.RegionSimilarityCalculator object.
      image_resizer_fn: a callable for image resizing.  This callable always
        takes a rank-3 image tensor (corresponding to a single image) and
        returns a rank-3 image tensor, possibly with new spatial dimensions.
        See builders/image_resizer_builder.py.
      non_max_suppression_fn: batch_multiclass_non_max_suppression
        callable that takes `boxes`, `scores` and optional `clip_window`
        inputs (with all other inputs already set) and returns a dictionary
        hold tensors with keys: `detection_boxes`, `detection_scores`,
        `detection_classes` and `num_detections`. See `post_processing.
        batch_multiclass_non_max_suppression` for the type and shape of these
        tensors.
      score_conversion_fn: callable elementwise nonlinearity (that takes tensors
        as inputs and returns tensors).  This is usually used to convert logits
        to probabilities.
      localization_loss_weight: float, same as lambda_coord as in the paper
      noobject_loss_weight: float, same as lambda_noobj as in the paper
      
    """
    super(YOLOMetaArch, self).__init__(num_classes=num_classes)
    self._is_training = is_training
    
    # Needed for fine-tuning from classification checkpoints whose
    # variables do not have the feature extractor scope.
    self._extract_features_scope = 'FeatureExtractor'

    self._feature_extractor = feature_extractor
    self._matcher = matcher
    self._region_similarity_calculator = region_similarity_calculator

    self._localization_loss_weight = localization_loss_weight
    self._noobject_loss_weight = noobject_loss_weight

    self._image_resizer_fn = image_resizer_fn
    self._non_max_suppression_fn = non_max_suppression_fn
    self._score_conversion_fn = score_conversion_fn

    self._add_summaries = add_summaries

  def preprocess(self, inputs):
    """Feature-extractor specific preprocessing.

    See base class.

    Args:
      inputs: a [batch, height_in, width_in, channels] float tensor representing
        a batch of images with values between 0 and 255.0.

    Returns:
      preprocessed_inputs: a [batch, height_out, width_out, channels] float
        tensor representing a batch of images.
    Raises:
      ValueError: if inputs tensor does not have type tf.float32
    """
    if inputs.dtype is not tf.float32:
      raise ValueError('`preprocess` expects a tf.float32 tensor')
    with tf.name_scope('Preprocessor'):
      resized_inputs = tf.map_fn(self._image_resizer_fn,
                                 elems=inputs,
                                 dtype=tf.float32)
      return self._feature_extractor.preprocess(resized_inputs)

  def predict(self, preprocessed_inputs):
    """Predicts unpostprocessed tensors from input tensor.

    This function takes an input batch of images and runs it through the forward
    pass of the network to yield unpostprocessesed predictions.

    Args:
      preprocessed_inputs: a [batch, height, width, channels] image tensor.
    Returns:
      prediction_dict: a dictionary holding prediction tensors with
        1) class_predictions : 4-D float tensor of shape [batch_size,
          grid_size * grid_size * boxes_per_cell, 1, num_classes] containing
           the conditional class probabilities for each grid cell
        2) box_scores : 4-D float tensor of shape [batch_size,
           grid_size * grid_size * boxes_per_cell, 2, 1] containing the confidence scores
           of each predicted box
        3) detection_boxes : 4-D float tensor of shape [batch_size,
          grid_size * grid_size * boxes_per_cell, 1, 4] containing the co-ordinates of
          the predicted bounding boxes
    """
    with tf.variable_scope(None, self._extract_features_scope,
                           [preprocessed_inputs]):
      feature_map = self._feature_extractor.extract_features(
          preprocessed_inputs)

      combined_shape = shape_utils.combined_static_and_dynamic_shape(feature_map)
      batch_size = combined_shape[0]
      boxes_per_cell = (combined_shape[-1] - self._num_classes) / 5

      # Extract the required values
      class_predictions = feature_map[:, :, :, 0 : self._num_classes]
      box_scores = feature_map[:, :, :, self._num_classes : self._num_classes + boxes_per_cell]
      detection_boxes = feature_map[:, :, :, self._num_classes + boxes_per_cell :]

      # These three variables have shapes [batch_size, grid_size, grid_size, X]
      # Reshape each of these to [batch_size, grid_size * grid_size * boxes_per_cell, X]
      class_predictions = tf.reshape(class_predictions, [batch_size, -1, 1, self.num_classes])
      box_scores = tf.reshape(box_scores, [batch_size, -1, 2, 1])
      detection_boxes = tf.reshape(detection_boxes, [batch_size, -1, 1, 4])

      # class, confidence scores, bounding box coordinates
      predictions_dict = {
          'class_predictions' : class_predictions,
          'box_scores' : box_scores,
          'detection_boxes' : detection_boxes,
      }
    return predictions_dict

  def postprocess(self, prediction_dict):
    """Converts prediction tensors to final detections.

    This function converts raw predictions tensors to final detection results by
    slicing off the background class, decoding box predictions and applying
    non max suppression and clipping to the image window.

    See base class for output format conventions.  Note also that by default,
    scores are to be interpreted as logits, but if a score_conversion_fn is
    used, then scores are remapped (and may thus have a different
    interpretation).

    Args:
      prediction_dict: a dictionary holding prediction tensors with
        1) class_predictions : 4-D float tensor of shape [batch_size,
          grid_size * grid_size * boxes_per_cell, 1, num_classes] containing the conditional class
          probabilities for each grid cell
        2) box_scores : 4-D float tensor of shape [batch_size,
          grid_size * grid_size * boxes_per_cell, 2, 1] containing the confidence scores of each predicted box
        3) detection_boxes : 4-D float tensor of shape [batch_size,
          grid_size * grid_size * boxes_per_cell, 1, 4] containing the co-ordinates of
          the predicted bounding boxes

    Returns:
      detections: a dictionary containing the following fields
        detection_boxes: [batch, max_detection, 4]
        detection_scores: [batch, max_detections]
        detection_classes: [batch, max_detections]
        num_detections: [batch]
    Raises:
      ValueError: if prediction_dict does not contain 'box_class_encodings' fields.
    """
    if ('class_predictons' not in prediction_dict) or ('box_scores' not in prediction_dict) \
            or ('detection_boxes' not in prediction_dict):
      raise ValueError('prediction_dict does not contain expected entries.')
    with tf.name_scope('Postprocessor'):

      class_predictions = prediction_dict['class_predictions']
      box_scores = prediction_dict['box_scores']
      detection_boxes = prediction_dict['detection_boxes']

      combined_shape = shape_utils.combined_static_and_dynamic_shape(class_predictions)
      batch_size = combined_shape[0]

      # multiply class conditional probabilities with box confidences
      class_predictions = tf.multiply(box_scores, class_predictions)
      # reshape class probabilities as required by non-max suppression
      class_predictions = tf.reshape(class_predictions, [batch_size, -1, self._num_classes])
      detection_scores = self._score_conversion_fn(class_predictions)
      clip_window = tf.constant([0, 0, 1, 1], tf.float32)
      (nmsed_boxes, nmsed_scores, nmsed_classes, _,
       num_detections) = self._non_max_suppression_fn(detection_boxes,
                                                      detection_scores,
                                                      clip_window=clip_window)
      return {'detection_boxes': nmsed_boxes,
              'detection_scores': nmsed_scores,
              'detection_classes': nmsed_classes,
              'num_detections': tf.to_float(num_detections)}

  def loss(self, prediction_dict, scope=None):
    """Compute scalar loss tensors with respect to provided groundtruth.

    Calling this function requires that groundtruth tensors have been
    provided via the provide_groundtruth function.

    Args:
      prediction_dict: a dictionary holding prediction tensors with
        1) class_predictions : 4-D float tensor of shape [batch_size,
          grid_size * grid_size * boxes_per_cell, 1, num_classes] containing the conditional class
          probabilities for each grid cell
        2) box_scores : 4-D float tensor of shape [batch_size,
          grid_size * grid_size * boxes_per_cell, 2, 1] containing the confidence scores of each predicted box
        3) detection_boxes : 4-D float tensor of shape [batch_size,
          grid_size * grid_size * boxes_per_cell, 1, 4] containing the co-ordinates of
          the predicted bounding boxes

    Returns:
      a dictionary mapping loss keys (`localization_loss`, `classification_loss`,
      `object loss` and 'noobj loss`) to scalar tensors representing corresponding
       loss values.
    """

    class_predictions = prediction_dict['class_predictions']
    box_scores = prediction_dict['box_scores']
    detection_boxes = prediction_dict['detection_boxes']

    with tf.name_scope(scope, 'Loss', prediction_dict.values()):
      I_ij_obj, I_i_obj, gt_detection_boxes, confidence, class_scores = self._assign_yolo_targets(
           self.groundtruth_lists(fields.BoxListFields.boxes),
           self.groundtruth_lists(fields.BoxListFields.classes),
           detection_boxes,)

      # TODO: I don't know what is summarize_input(), commenting it for now
      #if self._add_summaries:
      #  self._summarize_input(
      #      self.groundtruth_lists(fields.BoxListFields.boxes), match_list)

      # Take the square root of the widths and heights of the detected_boxes and merge
      # the tensor back using the concat operator
      detection_boxes = tf.concat([tf.expand_dims(detection_boxes[:, :, :, :2], 4),
                 tf.expand_dims(tf.sqrt(detection_boxes[:, :, :, 2]), 4),
                 tf.expand_dims(tf.sqrt(detection_boxes[:, :, :, 3]), 4)], 4)

      # compute the four individual terms for the loss function
      localization_loss = self._localization_loss_weight * I_ij_obj * tf.reduce_sum(
        tf.squared_difference(detection_boxes, gt_detection_boxes))

      object_loss = I_ij_obj * tf.reduce_sum(tf.squared_difference(
        box_scores, confidence))

      noobject_loss = self._noobject_loss_weight * (1 - I_ij_obj) * tf.reduce_sum(
        tf.squared_difference(box_scores, confidence))

      classification_loss = I_i_obj * tf.reduce_sum(tf.squared_difference(
        class_predictions, class_scores))


      # Not sure why we are supposed to return a dictionary
      # Not sure what the keys mean
      loss_dict = {
          'localization_loss': localization_loss,
          'classification_loss': classification_loss,
          'object_loss': object_loss,
          'noobject_loss':noobject_loss
      }

    return loss_dict

  def _assign_yolo_targets(self, groundtruth_boxes_list, groundtruth_classes_list, detection_boxes):
    """Assign groundtruth targets.

    Used to obtain regression and classification targets.

    Args:
      groundtruth_boxes_list: a list of 2-D tensors of shape [num_boxes, 4]
        containing coordinates of the groundtruth boxes.
          Groundtruth boxes are provided in [y_min, x_min, y_max, x_max]
          format and assumed to be normalized and clipped
          relative to the image window with y_min <= y_max and x_min <= x_max.
      groundtruth_classes_list: a list of 2-D one-hot (or k-hot) tensors of
        shape [num_boxes, num_classes] containing the class targets with the 0th
        index assumed to map to the first non-background class.
      detection_boxes: 4-D float tensor of shape [batch_size,
        grid_size * grid_size * boxes_per_cell, 1, 4] containing the co-ordinates of
        the predicted bounding boxes
    Returns:

      # TODO check the code below and replace the ? with appropraite values

      I_ij_obj: a tensor with shape [batch_size, ? ? ?]
      I_i_obj: a tensor with shape [batch_size, ? ? ?]
      detection_boxes: a tensor with shape [batch_size, ? ? ?]
      confidence: a tensor with shape [batch_size, ? ? ?]
      class_scores: a tensor with shape [batch_size, ? ? ?]
    """

    # TODO use tf.name_scope. Look in all the functions above

    # list of box lists where each box list contains ground truths for each image in a batch
    groundtruth_boxlists = [box_list.BoxList(boxes)
                            for boxes in groundtruth_boxes_list]

    # TODO: can we not hard code in the future? Yes, in the future
    S = 7  # grid size
    B = 2  # bounding boxes per grid cell
    image_size = 448  # image size

    # for each image create a list containing grid_size * grid_size
    # lists where each inner list is a list of ground truths of that
    # image in a particular grid cell
    # the final dimension of this list will be [batch_size, S*S, box_list_size]
    responsibility_list_batch = []
    responsibility_list_batch_classes = []

    # TODO Consider moving this chunk of a different function?
    for i in xrange(len(groundtruth_boxlists)):
      image_boxlist = groundtruth_boxlists[i]
      image_classlist = groundtruth_classes_list[i]

      grid_cell_responsibilities = [[] * (S * S)]
      grid_cell_responsibilities_classes = [[] * (S * S)]

      # Find the center of each ground truth bounding box and hence
      # the grid cell in which the ground truth lies
      ycenter, xcenter = image_boxlist.get_center_coordinates_and_sizes()[: 2]
      ycenter /= image_size
      ycenter *= S
      ycenter = tf.cast(ycenter, tf.int32)

      xcenter /= image_size
      xcenter *= S
      xcenter = tf.cast(xcenter, tf.int32)

      # grid_cell_index[i] is the grid cell in which the i'th ground truth is located
      grid_cell_index = xcenter * S + ycenter

      # list of tensor objects corresponding to each ground truth box
      boxes = tf.unstack(image_boxlist.get())
      classes = tf.unstack(image_classlist)

      # push each ground truth and its corresponding one-hot encoding
      # into the correct place positions
      num_boxes = len(boxes)
      for i in xrange(num_boxes):
        grid_cell_responsibilities[grid_cell_index[i]].append(boxes[i])
        grid_cell_responsibilities_classes[grid_cell_index[i]].append(classes[i])

      # convert each list of tensors into a box_list.BoxList
      for i in xrange(S * S):
        grid_cell_responsibilities[i] = box_list.BoxList(tf.stack(grid_cell_responsibilities[i]))

      # append the list for this batch into the final list
      responsibility_list_batch.append(grid_cell_responsibilities)
      responsibility_list_batch_classes.append(grid_cell_responsibilities_classes)


    # Create a list similar to responsibility_list but for ground truth boxes
    # the idea is that for each batch b and for each grid cell i, we will apply
    # the matcher on the similarity matrix of prediction_box_list_batch[batch_num][i]
    # and responsibility_list_batch[batch_num][i] and use this to obtain the indicators
    # I_ij_obj etc

    # the final dimension of this list should be [batch_size, S*S, box_list_size]
    prediction_box_list_batch = []
    detection_boxes_unstacked = tf.unstack(detection_boxes)

    # xoffets and yoffsets correspond to the top left corners of each grid cell
    xoffsets = tf.constant([[i * image_size / S]  for i in xrange(S)
                            for j in xrange(S) for k in xrange(2)])
    yoffsets = tf.constant([[j * image_size / S]  for i in xrange(S)
                            for j in xrange(S) for k in xrange(2)])

    # TODO Consider moving this chunk to another function?
    for plist in detection_boxes_unstacked:
      # convert the predicted bounding boxes into the format
      # [ymin, xmin, ymax, xmax] so that they can be converted into a box_list.BoxList

      # As per the paper the predicted coordaintes have the following interpretation
      # x, y = center of predicted box relative to grid cell
      # w, h of bounding box are normalized wrt width and height of the image
      xcenter = plist[:, :, 0] * 64 + xoffsets
      ycenter = plist[:, :, 1] * 64 + yoffsets
      w = plist[:, :, 2] * image_size
      h = plist[:, :, 3] * image_size

      xmin = xcenter - h * 0.5
      ymin = ycenter - w * 0.5
      xmax = xcenter + h * 0.5
      ymax = ycenter + w * 0.5

      # combine the tensors into a single tensor
      plist = tf.concat([ymin, xmin, ymax, xmax], 1)

      # convert to a BoxList and append to appropriate to prediction_box_list_batch
      unstacked_plist = tf.unstack(plist)
      prediction_box_list = [box_list.BoxList(boxes) for boxes in unstacked_plist]
      
      assert len(prediction_box_list) == S * S

      prediction_box_list_batch.append(prediction_box_list)

    # sanity check, batch sizes of the two lists should be the same
    assert len(responsibility_list_batch) == len(prediction_box_list_batch)
    num_batches = len(responsibility_list_batch)

    # These lists will be cast into tensors and then returned as required by the function
    I_ij_obj_list_batch = []
    I_i_obj_list_batch = []
    detection_boxes_ground_truth_batch = []
    confidence_ground_truth_batch = []
    class_probability_ground_truth_batch = []

    # TODO Consider moving this chunk to another function
    for batch_num in xrange(num_batches):
      # create a list of 0s and 1s corresponding to
      # I_ij_obj etc. for each image in the batch
      I_ij_obj_list = []
      I_i_obj_list = []
      detection_boxes_ground_truth_list = []
      confidence_ground_truth_list = []
      class_probability_ground_truth_list = []

      # compute the similarity matrix for each grid cell
      # and pass it to the matcher object to match predicted bounding boxes
      # with the ground truths that must be predicted from this cell
      for i in xrange(S*S):
        list1 = prediction_box_list_batch[batch_num][i]
        list2 = responsibility_list_batch[batch_num][i]
        one_hots = responsibility_list_batch_classes[batch_num][i]

        # corner case, if there is no ground truth in this cell
        # passing an empty list to the Matcher class will raise an error
        if len(list2) == 0:
          # By setting these values to 0 we ensure that no matter what the
          # result of the computation is in X - \hat(X), it will get multiplied by 0
          # and won't have any contribution in the answer

          # Two values for I_ij_obj, one multiplier for each predicted bounding box
          I_ij_obj_list.append(tf.constant([[0]]))
          I_ij_obj_list.append(tf.constant([[0]]))

          # Only one value is required for I_i_obj
          I_i_obj_list.append(tf.constant([[0]]))

          # now add dummy data to ensure that other tensors have the correct size
          detection_boxes_ground_truth_list.append(tf.constant([[0, 0, 0, 0]]))
          detection_boxes_ground_truth_list.append(tf.constant([[0, 0, 0, 0]]))
          confidence_ground_truth_list.append(tf.constant([[0]]))
          class_probability_ground_truth_list.append(tf.constant([[0] * self._num_classes]))
          continue

        # create the match object using the provided Matcher
        matchObject = self._matcher(self._region_similarity_calculator(list1, list2))


        # 0 if no ground truth was matched, 1 otherwise
        # remember that the columns in the similarity matrix corresponded to the
        # ground truth bounding boxes
        tmp_columns = tf.unstack(matchObject.matched_column_indicator())
        for col_id in tmp_columns:
          I_ij_obj_list.append(tf.constant([col_id]))

        # if there was at least one ground truth that was matched to some prediction,
        # then we append 1, else we append 0
        I_i_obj_list.append(tf.constant([[1 if len(tmp_columns) > 0 else 0]]))

        # now we must convert the ground truth coordinates from
        # ymin ymax xmin xmax to xc, yc, w, h
        # i.e., center of box relative to grid cell
        # and width and height scaled by image width and height
        # we do this because in the computation of the loss function
        # we intend to do X - X_gt and hence X_gt must be in the same format as X
        # esentially, we have just followed the paper
        tly = i % S
        tlx = i / S
        window = [tly * 64, tlx * 64, (tly+1) * 64, (tlx+1) * 64]
        h, w = box_list_ops.height_width(list2)

        # TODO pretty long line :/
        ycenter, xcenter = box_list_ops.change_coordinate_frame(list2, tf.constant(window)).get_center_coordinates_and_sizes()[: 2]

        xcenter = tf.expand_dims(xcenter, 1)
        ycenter = tf.expand_dims(ycenter, 1)
        w = tf.expand_dims(tf.sqrt(w), 1)
        h = tf.expand_dims(tf.sqrt(h), 1)

        # this corresponds to the ground truth data after it has been
        normalized_gt_boxes = tf.concat([xcenter, ycenter, w, h], 1)

        # if there was no match for a particular predicted bounding box,
        # then we can just use the co-ordinates of any gt bounding box
        # this is because no matter what the value is, I_ij_obj will ensure
        # that the contribution from this location would sum to 0
        match_results = matchObject.match_results()
        match_results = tf.where(tf.greater(match_results, -1), match_results, tf.zeros_like(match_results))

        # build the X_gt list by placing the coordinates of the matched gt box
        # at the correct location
        for gbox in tf.unstack(tf.expand_dims(tf.gather(normalized_gt_boxes, match_results), 1)):
          detection_boxes_ground_truth_list.append(gbox)
          confidence_ground_truth_list.append(tf.constant([[1]]))

        class_probability_ground_truth_list.append(tf.expand_dims(tf.gather(one_hots, match_results[0]), 1))

      # append all the lists for one image to the lists corresponding to the batches
      I_ij_obj_list_batch.append(I_ij_obj_list)
      I_i_obj_list_batch.append(I_i_obj_list)
      detection_boxes_ground_truth_batch.append(detection_boxes_ground_truth_list)
      confidence_ground_truth_batch.append(confidence_ground_truth_list)
      class_probability_ground_truth_batch.append(class_probability_ground_truth_list)

    # convert the lists into Tensors and return
    I_ij_obj = tf.stack(I_ij_obj_list_batch)
    I_i_obj = tf.stack(I_i_obj_list_batch)
    detection_boxes = tf.stack(detection_boxes_ground_truth_batch)
    confidence = tf.stack(confidence_ground_truth_batch)
    class_scores = tf.stack(class_probability_ground_truth_batch)

    return I_ij_obj, I_i_obj, detection_boxes, confidence, class_scores

  def restore_map(self, from_detection_checkpoint=True):
    """Returns a map of variables to load from a foreign checkpoint.

    See parent class for details.

    Args:
      from_detection_checkpoint: whether to restore from a full detection
        checkpoint (with compatible variable names) or to restore from a
        classification checkpoint for initialization prior to training.

    Returns:
      A dict mapping variable names (to load from a checkpoint) to variables in
      the model graph.
    """
    variables_to_restore = {}
    for variable in tf.all_variables():
      if variable.op.name.startswith(self._extract_features_scope):
        var_name = variable.op.name
        if not from_detection_checkpoint:
          var_name = (re.split('^' + self._extract_features_scope + '/',
                               var_name)[-1])
        variables_to_restore[var_name] = variable
    return variables_to_restore

