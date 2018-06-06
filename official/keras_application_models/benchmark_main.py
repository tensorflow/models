# import the necessary packages
import os

# pylint: disable=g-bad-import-order
import numpy as np
from absl import app as absl_app
from absl import flags
import tensorflow as tf
# pylint: enable=g-bad-import-order

from official.utils.flags import core as flags_core
from official.utils.logs import logger

# import official.keras_application_models.model_callbacks as model_callbacks
import model_callbacks
import official.resnet.imagenet_main as imagenet_main
import official.resnet.imagenet_preprocessing as imagenet_preprocessing
import official.resnet.resnet_run_loop as resnet_run_loop

_NUM_CHANNELS = 3
_NUM_CLASSES = 1000

# Define a dictionary that maps model names to their model classes inside Keras
MODELS = {
  "vgg16": tf.keras.applications.VGG16,
  "vgg19": tf.keras.applications.VGG19,
  "inception": tf.keras.applications.InceptionV3,
  "xception": tf.keras.applications.Xception, # TensorFlow ONLY
  "resnet": tf.keras.applications.ResNet50,
  "inceptionresnet": tf.keras.applications.InceptionResNetV2,
  "mobilenet": tf.keras.applications.MobileNet,
  "densenet121": tf.keras.applications.DenseNet121,
  "densenet169": tf.keras.applications.DenseNet169,
  "densenet201": tf.keras.applications.DenseNet201,
  "nasnetlarge": tf.keras.applications.NASNetLarge,
  "nasnetmobile": tf.keras.applications.NASNetMobile,
}


def get_model_callbacks(
    benchmark_level, callbacks_list, batch_size=None, epoch_size=None,
    metrics=None, metric_logger=None):
  if not callbacks_list:
    return []

  batch_based = True if benchmark_level == "batch_based" else None
  epoch_based = True if benchmark_level == "epoch_based" else None

  callbacks = []
  for callback in callbacks_list:
    callback_name = callback.strip().lower()
    if callback_name not in model_callbacks.CALLBACKS:
      raise ValueError('Unrecognized callback requested: {}'.format(callback))

    if callback_name == model_callbacks.CALLBACKS[0]:
      exp_per_second_callback = model_callbacks.ExamplesPerSecondCallback(
          batch_size=batch_size, epoch_size=epoch_size,
          batch_based=batch_based, epoch_based=epoch_based,
          metric_logger=metric_logger)
      callbacks.append(exp_per_second_callback)
    else:
      logging_metric_callback = model_callbacks.LoggingMetricCallback(
          metrics=metrics, metric_logger=metric_logger,
          batch_based=batch_based, epoch_based=epoch_based)
      callbacks.append(logging_metric_callback)

  return callbacks


def generate_synthetic_input_dataset(image_size, batch_size):
  # Generate synthetic dataset
  input_shape = (batch_size, ) + image_size + (_NUM_CHANNELS, )

  images = tf.zeros(input_shape, dtype=tf.float32)
  labels = tf.zeros((batch_size, _NUM_CLASSES), dtype=tf.float32)

  return tf.data.Dataset.from_tensors((images, labels)).repeat()

def get_default_image_size(model):
  """Provide default image size for each model. """
  image_size = (224, 224)
  if model in ("inception", "xception", "inceptionresnet"):
    image_size = (299, 299)
  elif model in ("nasnetlarge"):
    image_size = (331, 331)
  return image_size


def main(_):
  # Ensure a valid model name was supplied via command line argument
  if FLAGS.model not in MODELS.keys():
    raise AssertionError("The --model command line argument should "
                         "be a key in the `MODELS` dictionary.")

  # Load the model
  tf.logging.info("Benchmark on {} model...".format(FLAGS.model))
  Network = MODELS[FLAGS.model]
  model = Network(weights=None)

  image_size = get_default_image_size(FLAGS.model)
  if FLAGS.use_synthetic_data:
    tf.logging.info("Using synthetic dataset...")
    batch_size=resnet_run_loop.per_device_batch_size(
        FLAGS.batch_size, flags_core.get_num_gpus(FLAGS))
    train_dataset = generate_synthetic_input_dataset(image_size, batch_size)
    val_dataset = generate_synthetic_input_dataset(image_size, batch_size)
  else:
    # Use the actual ImageNet dataset (TODO)
    pass

  # If run with multiple GPUs
  num_gpus = flags_core.get_num_gpus(FLAGS)
  if num_gpus > 0:
    model = tf.keras.utils.multi_gpu_model(model, gpus=num_gpus)

  # Configure the model using the optimizer sgd
  model.compile(loss="categorical_crossentropy",
                optimizer="sgd",
                metrics=["accuracy"])

  run_params = {
      "batch_size": FLAGS.batch_size,
      "synthetic_data": FLAGS.use_synthetic_data,
      "benchmark_level": FLAGS.benchmark_level,
      "train_epochs": FLAGS.train_epochs
  }
  benchmark_logger = logger.config_benchmark_logger(FLAGS)
  benchmark_logger.log_run_info(
      model_name=FLAGS.model,
      dataset_name="ImageNet",
      run_params=run_params)

  # Create callbacks
  if FLAGS.use_synthetic_data:
    train_num_images = batch_size
    val_num_images = batch_size

  callbacks = get_model_callbacks(
      FLAGS.benchmark_level, FLAGS.callbacks,
      batch_size=FLAGS.batch_size, epoch_size=train_num_images,
      metrics=None, metric_logger=benchmark_logger)

  history = model.fit(
      train_dataset,
      epochs=FLAGS.train_epochs,
      callbacks=callbacks,
      validation_data=val_dataset,
      steps_per_epoch=int(np.ceil(train_num_images / float(batch_size))),
      validation_steps=int(np.ceil(val_num_images / float(batch_size)))
  )

  tf.logging.info("Logging the evaluation results...")
  for epoch in range(FLAGS.train_epochs):
    eval_results = {
        "val_acc": history.history["val_acc"][epoch],
        "val_loss": history.history["val_loss"][epoch],
        "epoch": epoch + 1,
        tf.GraphKeys.GLOBAL_STEP: (epoch + 1) * np.ceil(train_num_images/batch_size)
    }
    benchmark_logger.log_evaluation_result(eval_results)

  # Clear the session explicitly to avoid session delete error
  tf.keras.backend.clear_session()


def define_keras_benchmark_flags():
  """Add flags for keras built-in application models."""
  flags_core.define_base()
  flags_core.define_performance()
  flags_core.define_image()
  flags_core.define_benchmark()
  flags.adopt_module_key_flags(flags_core)

  flags_core.set_defaults(
    data_format="channels_last",
    use_synthetic_data=True,
    hooks=False,
    train_epochs=2)

  flags.DEFINE_enum(
    name="model", default="resnet",
    enum_values=MODELS.keys(), case_sensitive=False,
    help=flags_core.help_wrap(
        "Model to be benchmarked."))

  flags.DEFINE_enum(
    name="benchmark_level", default="epoch_based",
    enum_values=["epoch_based", "batch_based"], case_sensitive=False,
    help=flags_core.help_wrap(
        "Which level to benchmark the training process."))

  flags.DEFINE_list(
      name="callbacks",
      default=["ExamplesPerSecondCallback", "LoggingMetricCallback"],
      help=flags_core.help_wrap(
          "A list of (case insensitive) strings to specify the names of "
          "callbacks. For example: `--callbacks ExamplesPerSecondCallback,"
          "LoggingMetricCallback`"))


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  define_keras_benchmark_flags()
  FLAGS = flags.FLAGS
  absl_app.run(main)
