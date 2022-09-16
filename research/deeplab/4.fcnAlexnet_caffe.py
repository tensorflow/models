# data layers
layer {
  name: "data"
  type: "Data"
  top: "data"
  include {
    phase: TRAIN
  }
  data_param {
    batch_size: 1
    backend: LMDB
  }
}
layer {
  name: "label"
  type: "Data"
  top: "label"
  include {
    phase: TRAIN
  }
  data_param {
    batch_size: 1
    backend: LMDB
  }
}
layer {
  name: "data"
  type: "Data"
  top: "data"
  include {
    phase: TEST
  }
  data_param {
    batch_size: 1
    backend: LMDB
  }
}
layer {
  name: "label"
  type: "Data"
  top: "label"
  include {
    phase: TEST
  }
  data_param {
    batch_size: 1
    backend: LMDB
  }
}
# data preprocessing
layer {
  # Use Power layer in deploy phase for input scaling
  name: "shift"
  bottom: "data"
  top: "data_preprocessed"
  type: "Power"
  power_param {
    shift: -116.0
  }
}
# main network description
layer {
  name: "conv1"
  type: "Convolution"
  bottom: "data_preprocessed"
  top: "conv1"
  convolution_param {
    num_output: 96
    pad: 100
    kernel_size: 11
    group: 1
    stride: 4
  }
}
layer {
  name: "relu1"
  type: "ReLU"
  bottom: "conv1"
  top: "conv1"
}
layer {
  name: "pool1"
  type: "Pooling"
  bottom: "conv1"
  top: "pool1"
  pooling_param {
    pool: MAX
    kernel_size: 3
    stride: 2
  }
}
layer {
  name: "norm1"
  type: "LRN"
  bottom: "pool1"
  top: "norm1"
  lrn_param {
    local_size: 5
    alpha: 0.0001
    beta: 0.75
  }
}
layer {
  name: "conv2"
  type: "Convolution"
  bottom: "norm1"
  top: "conv2"
  convolution_param {
    num_output: 256
    pad: 2
    kernel_size: 5
    group: 2
    stride: 1
  }
}
layer {
  name: "relu2"
  type: "ReLU"
  bottom: "conv2"
  top: "conv2"
}
layer {
  name: "pool2"
  type: "Pooling"
  bottom: "conv2"
  top: "pool2"
  pooling_param {
    pool: MAX
    kernel_size: 3
    stride: 2
  }
}
layer {
  name: "norm2"
  type: "LRN"
  bottom: "pool2"
  top: "norm2"
  lrn_param {
    local_size: 5
    alpha: 0.0001
    beta: 0.75
  }
}
layer {
  name: "conv3"
  type: "Convolution"
  bottom: "norm2"
  top: "conv3"
  convolution_param {
    num_output: 384
    pad: 1
    kernel_size: 3
    group: 1
    stride: 1
  }
}
layer {
  name: "relu3"
  type: "ReLU"
  bottom: "conv3"
  top: "conv3"
}
layer {
  name: "conv4"
  type: "Convolution"
  bottom: "conv3"
  top: "conv4"
  convolution_param {
    num_output: 384
    pad: 1
    kernel_size: 3
    group: 2
    stride: 1
  }
}
layer {
  name: "relu4"
  type: "ReLU"
  bottom: "conv4"
  top: "conv4"
}
layer {
  name: "conv5"
  type: "Convolution"
  bottom: "conv4"
  top: "conv5"
  convolution_param {
    num_output: 256
    pad: 1
    kernel_size: 3
    group: 2
    stride: 1
  }
}
layer {
  name: "relu5"
  type: "ReLU"
  bottom: "conv5"
  top: "conv5"
}
layer {
  name: "pool5"
  type: "Pooling"
  bottom: "conv5"
  top: "pool5"
  pooling_param {
    pool: MAX
    kernel_size: 3
    stride: 2
  }
}
layer {
  name: "fc6"
  type: "Convolution"
  bottom: "pool5"
  top: "fc6"
  convolution_param {
    num_output: 4096
    pad: 0
    kernel_size: 6
    group: 1
    stride: 1
  }
}
layer {
  name: "relu6"
  type: "ReLU"
  bottom: "fc6"
  top: "fc6"
}
layer {
  name: "drop6"
  type: "Dropout"
  bottom: "fc6"
  top: "fc6"
  dropout_param {
    dropout_ratio: 0.5
  }
}
layer {
  name: "fc7"
  type: "Convolution"
  bottom: "fc6"
  top: "fc7"
  convolution_param {
    num_output: 4096
    pad: 0
    kernel_size: 1
    group: 1
    stride: 1
  }
}
layer {
  name: "relu7"
  type: "ReLU"
  bottom: "fc7"
  top: "fc7"
}
layer {
  name: "drop7"
  type: "Dropout"
  bottom: "fc7"
  top: "fc7"
  dropout_param {
    dropout_ratio: 0.5
  }
}
layer {
  name: "score_frNew"
  type: "Convolution"
  bottom: "fc7"
  top: "score_frNew"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 4
    pad: 0
    kernel_size: 1
  }
}
layer {
  name: "upscoreNew"
  type: "Deconvolution"
  bottom: "score_frNew"
  top: "upscoreNew"
  param {
    lr_mult: 0
  }
  convolution_param {
    num_output: 4
    group: 4
    bias_term: false
    kernel_size: 63
    stride: 32
    weight_filler: { type: "bilinear" }
  }
}
layer {
  name: "score"
  type: "Crop"
  bottom: "upscoreNew"
  bottom: "data"
  top: "score"
  crop_param {
    axis: 2
    offset: 18
  }
}
layer {
  name: "loss"
  type: "SoftmaxWithLoss"
  bottom: "score"
  bottom: "label"
  top: "loss"
  loss_param {
    ignore_label: 255
    normalize: true
  }
  exclude {
    stage: "deploy"
  }
}
layer {
  name: "accuracy"
  type: "Accuracy"
  bottom: "score"
  bottom: "label"
  top: "accuracy"
  include { stage: "val" }
  accuracy_param { ignore_label: 255 }
}