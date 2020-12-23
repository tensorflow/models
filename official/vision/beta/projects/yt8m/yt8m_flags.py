

from absl import flags


def define_flags():
  """Defines flags."""
  flags.DEFINE_string(
      'experiment', default='yt8m_experiment', help='The experiment type registered.')

  flags.DEFINE_enum(
      'mode',
      default=None,
      enum_values=['train', 'eval', 'train_and_eval',
                   'continuous_eval', 'continuous_train_and_eval'],
      help='Mode to run: `train`, `eval`, `train_and_eval`, '
      '`continuous_eval`, and `continuous_train_and_eval`.')

  flags.DEFINE_string(
      'model_dir',
      default=None,
      help='The directory where the model and training/evaluation summaries'
      'are stored.')

  flags.DEFINE_multi_string(
      'config_file',
      default=None,
      help='YAML/JSON files which specifies overrides. The override order '
      'follows the order of args. Note that each file '
      'can be used as an override template to override the default parameters '
      'specified in Python. If the same parameter is specified in both '
      '`--config_file` and `--params_override`, `config_file` will be used '
      'first, followed by params_override.')

  flags.DEFINE_string(
      'params_override',
      default=None,
      help='a YAML/JSON string or a YAML file which specifies additional '
      'overrides over the default parameters and those specified in '
      '`--config_file`. Note that this is supposed to be used only to override '
      'the model parameters, but not the parameters like TPU specific flags. '
      'One canonical use case of `--config_file` and `--params_override` is '
      'users first define a template config file using `--config_file`, then '
      'use `--params_override` to adjust the minimal set of tuning parameters, '
      'for example setting up different `train_batch_size`. The final override '
      'order of parameters: default_model_params --> params from config_file '
      '--> params in params_override. See also the help message of '
      '`--config_file`.')

  flags.DEFINE_multi_string(
      'gin_file', default=None, help='List of paths to the config files.')

  flags.DEFINE_multi_string(
      'gin_params',
      default=None,
      help='Newline separated list of Gin parameter bindings.')

  flags.DEFINE_string(
      'tpu', default=None,
      help='The Cloud TPU to use for training. This should be either the name '
      'used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 '
      'url.')

  flags.DEFINE_string(
      'tf_data_service', default=None, help='The tf.data service address')


#yt8m flags
  flags.DEFINE_string("train_dir", "/tmp/yt8m_model/",
                      "The directory to save the model files in.")
  flags.DEFINE_string(
      "input_path", "",
      "File glob for the training dataset. If the files refer to Frame Level "
      "features (i.e. tensorflow.SequenceExample), then set --reader_type "
      "format. The (Sequence)Examples are expected to have 'rgb' byte array "
      "sequence feature as well as a 'labels' int64 context feature.")
  flags.DEFINE_string("feature_names", "mean_rgb", "Name of the feature "
                      "to use for training.")
  flags.DEFINE_string("feature_sizes", "1024", "Length of the feature vectors.")

  # Model flags.
  flags.DEFINE_bool(
      "frame_features", False,
      "If set, then --train_input_path must be frame-level features. "
      "Otherwise, --train_input_path must be aggregated video-level "
      "features. The model must also be set appropriately (i.e. to read 3D "
      "batches VS 4D batches.")
  flags.DEFINE_bool(
      "segment_labels", False,
      "If set, then --train_input_path must be frame-level features (but with"
      " segment_labels). Otherwise, --train_input_path must be aggregated "
      "video-level features. The model must also be set appropriately (i.e. to "
      "read 3D batches VS 4D batches.")
  flags.DEFINE_string(
      "model", "LogisticModel",
      "Which architecture to use for the model. Models are defined "
      "in models.py.")
  flags.DEFINE_bool(
      "start_new_model", False,
      "If set, this will not resume from a checkpoint and will instead create a"
      " new model instance.")

  # Training flags.
  flags.DEFINE_integer(
      "num_gpu", 1, "The maximum number of GPU devices to use for training. "
      "Flag only applies if GPUs are installed")
  flags.DEFINE_integer("batch_size", 1024,
                       "How many examples to process per batch for training.")
  flags.DEFINE_string("label_loss", "CrossEntropyLoss",
                      "Which loss function to use for training the model.")
  flags.DEFINE_float(
      "regularization_penalty", 1.0,
      "How much weight to give to the regularization loss (the label loss has "
      "a weight of 1).")
  flags.DEFINE_float("base_learning_rate", 0.01,
                     "Which learning rate to start with.")
  flags.DEFINE_float(
      "learning_rate_decay", 0.95,
      "Learning rate decay factor to be applied every "
      "learning_rate_decay_examples.")
  flags.DEFINE_float(
      "learning_rate_decay_examples", 4000000,
      "Multiply current learning rate by learning_rate_decay "
      "every learning_rate_decay_examples.")
  flags.DEFINE_integer(
      "num_epochs", 5, "How many passes to make over the dataset before "
      "halting training.")
  flags.DEFINE_integer(
      "max_steps", None,
      "The maximum number of iterations of the training loop.")
  flags.DEFINE_integer(
      "export_model_steps", 1000,
      "The period, in number of steps, with which the model "
      "is exported for batch prediction.")

  # Other flags.
  flags.DEFINE_integer("num_readers", 8,
                       "How many threads to use for reading input files.")
  flags.DEFINE_string("optimizer", "AdamOptimizer",
                      "What optimizer class to use.")
  flags.DEFINE_float("clip_gradient_norm", 1.0, "Norm to clip gradients to.")
  flags.DEFINE_bool(
    "log_device_placement", False,
    "Whether to write the device on which every op will run into the "
    "logs on startup.")


  # Eval flags
  # flags.DEFINE_string(
  #   "eval_input_path", "",
  #   "File glob defining the evaluation dataset in tensorflow.SequenceExample "
  #   "format. The SequenceExamples are expected to have an 'rgb' byte array "
  #   "sequence feature as well as a 'labels' int64 context feature.")
  flags.DEFINE_boolean("run_once", False, "Whether to run eval only once.")
  flags.DEFINE_integer("top_k", 20, "How many predictions to output per video.")

