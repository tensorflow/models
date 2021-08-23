from official.vision.beta.modeling.backbones import factory
from yolo.modeling.decoders.yolo_decoder import YoloDecoder
from yolo.modeling.heads.yolo_head import YoloHead
from yolo.modeling.layers.detection_generator import YoloLayer
from yolo.modeling.backbones.darknet import build_darknet

from yolo.modeling import yolo_model
from yolo.configs import yolo


def build_yolo_decoder(input_specs, model_config: yolo.Yolo, l2_regularization):
  activation = (
      model_config.decoder.activation
      if model_config.decoder.activation != "same" else
      model_config.norm_activation.activation)
  subdivisions = 1

  if model_config.decoder.version is None:  # custom yolo
    model = YoloDecoder(
        input_specs,
        embed_spp=model_config.decoder.embed_spp,
        use_fpn=model_config.decoder.use_fpn,
        fpn_depth=model_config.decoder.fpn_depth,
        path_process_len=model_config.decoder.path_process_len,
        max_level_process_len=model_config.decoder.max_level_process_len,
        xy_exponential=model_config.decoder.xy_exponential,
        activation=activation,
        subdivisions=subdivisions,
        use_spatial_attention=model_config.use_sam,
        use_sync_bn=model_config.norm_activation.use_sync_bn,
        norm_momentum=model_config.norm_activation.norm_momentum,
        norm_epsilon=model_config.norm_activation.norm_epsilon,
        kernel_regularizer=l2_regularization)
    return model

  if model_config.decoder.type == None:
    model_config.decoder.type = "regular"

  if model_config.decoder.version not in yolo_model.YOLO_MODELS.keys():
    raise Exception(
        "unsupported model version please select from {v3, v4}, \n\n \
        or specify a custom decoder config using YoloDecoder in you yaml")

  if model_config.decoder.type not in yolo_model.YOLO_MODELS[
      model_config.decoder.version].keys():
    raise Exception("unsupported model type please select from \
        {yolo_model.YOLO_MODELS[model_config.decoder.version].keys()},\
        \n\n or specify a custom decoder config using YoloDecoder in you yaml")

  base_model = yolo_model.YOLO_MODELS[model_config.decoder.version][
      model_config.decoder.type]

  cfg_dict = model_config.decoder.as_dict()
  for key in base_model:
    if cfg_dict[key] is not None:
      base_model[key] = cfg_dict[key]

  base_dict = dict(
      activation=activation,
      subdivisions=subdivisions,
      use_spatial_attention=model_config.decoder.use_spatial_attention,
      use_sync_bn=model_config.norm_activation.use_sync_bn,
      norm_momentum=model_config.norm_activation.norm_momentum,
      norm_epsilon=model_config.norm_activation.norm_epsilon,
      kernel_regularizer=l2_regularization)

  base_model.update(base_dict)
  print(base_model)

  model = YoloDecoder(input_specs, **base_model)

  return model


def build_yolo_filter(model_config: yolo.Yolo, decoder: YoloDecoder, masks,
                      xy_scales, path_scales):

  def _build(values):
    print(values)
    if "all" in values and values["all"] is not None:
      for key in values:
        if key != 'all':
          values[key] = values["all"]
    print(values)
    return values

  model = YoloLayer(
      masks=masks,
      classes=model_config.num_classes,
      anchors=model_config._boxes,
      iou_thresh=model_config.filter.iou_thresh,
      nms_thresh=model_config.filter.nms_thresh,
      max_boxes=model_config.filter.max_boxes,
      nms_type=model_config.filter.nms_type,
      path_scale=path_scales,
      scale_xy=xy_scales,
      darknet=model_config.filter.darknet, 
      label_smoothing=model_config.filter.label_smoothing,
      pre_nms_points=model_config.filter.pre_nms_points,
      use_scaled_loss=model_config.filter.use_scaled_loss,
      truth_thresh=_build(model_config.filter.truth_thresh.as_dict()),
      loss_type=_build(model_config.filter.loss_type.as_dict()),
      max_delta=_build(model_config.filter.max_delta.as_dict()),
      new_cords=_build(model_config.filter.new_cords.as_dict()),
      iou_normalizer=_build(model_config.filter.iou_normalizer.as_dict()),
      cls_normalizer=_build(model_config.filter.cls_normalizer.as_dict()),
      obj_normalizer=_build(model_config.filter.obj_normalizer.as_dict()),
      ignore_thresh=_build(model_config.filter.ignore_thresh.as_dict()),
      objectness_smooth=_build(model_config.filter.objectness_smooth.as_dict()))
  return model


def build_yolo_head(input_specs, model_config: yolo.Yolo, l2_regularization):
  head = YoloHead(
      min_level=model_config.min_level,
      max_level=model_config.max_level,
      classes=model_config.num_classes,
      boxes_per_level=model_config.boxes_per_scale,
      norm_momentum=model_config.norm_activation.norm_momentum,
      norm_epsilon=model_config.norm_activation.norm_epsilon,
      kernel_regularizer=l2_regularization, 
      smart_bias=model_config.smart_bias)
  return head


def build_yolo(input_specs, model_config, l2_regularization, masks, xy_scales,
               path_scales):
  print(model_config.as_dict())
  print(input_specs)
  print(l2_regularization)

  # backbone = factory.build_backbone(input_specs, model_config,
  #                                   l2_regularization)
  backbone = build_darknet(input_specs, model_config,
                                    l2_regularization)
  decoder = build_yolo_decoder(backbone.output_specs, model_config,
                               l2_regularization)
  head = build_yolo_head(decoder.output_specs, model_config, l2_regularization)
  filter = build_yolo_filter(model_config, head, masks, xy_scales, path_scales)

  model = yolo_model.Yolo(
      backbone=backbone, decoder=decoder, head=head, filter=filter)
  model.build(input_specs.shape)
  model.decoder.summary()
  model.summary()

  losses = filter.losses
  return model, losses
