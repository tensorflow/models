

from absl import flags

def define_flags():
  """Defines flags."""
  flags.DEFINE_boolean("run_once", False, "Whether to run eval only once.")

  # Training flags. -> add trainer
  # flags.DEFINE_float(
  #     "regularization_penalty", 1.0,
  #     "How much weight to give to the regularization loss (the label loss has "
  #     "a weight of 1).")
  flags.DEFINE_integer(
      "max_steps", None,
      "The maximum number of iterations of the training loop.")
  flags.DEFINE_integer(
      "export_model_steps", 1000,
      "The period, in number of steps, with which the model "
      "is exported for batch prediction.")

  # Other flags. -> task 바로 밑에
  flags.DEFINE_float("clip_gradient_norm", 1.0, "Norm to clip gradients to.")
  flags.DEFINE_bool(
    "log_device_placement", False,
    "Whether to write the device on which every op will run into the "
    "logs on startup.")

