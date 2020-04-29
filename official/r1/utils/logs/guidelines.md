# Logging in official models

This library adds logging functions that print or save tensor values. Official models should define all common hooks
(using hooks helper) and a benchmark logger.

1. **Training Hooks**

   Hooks are a TensorFlow concept that define specific actions at certain points of the execution. We use them to obtain and log
   tensor values during training.

   hooks_helper.py provides an easy way to create common hooks. The following hooks are currently defined:
   * LoggingTensorHook: Logs tensor values
   * ProfilerHook: Writes a timeline json that can be loaded into chrome://tracing.
   * ExamplesPerSecondHook: Logs the number of examples processed per second.
   * LoggingMetricHook: Similar to LoggingTensorHook, except that the tensors are logged in a format defined by our data
     anaylsis pipeline.


2. **Benchmarks**

   The benchmark logger provides useful functions for logging environment information, and evaluation results.
   The module also contains a context which is used to update the status of the run.

Example usage:

```
from absl import app as absl_app

from official.utils.logs import hooks_helper
from official.utils.logs import logger

def model_main(flags_obj):
  estimator = ...

  benchmark_logger = logger.get_benchmark_logger()
  benchmark_logger.log_run_info(...)

  train_hooks = hooks_helper.get_train_hooks(...)

  for epoch in range(10):
    estimator.train(..., hooks=train_hooks)
    eval_results = estimator.evaluate(...)

    # Log a dictionary of metrics
    benchmark_logger.log_evaluation_result(eval_results)

    # Log an individual metric
    benchmark_logger.log_metric(...)


def main(_):
  with logger.benchmark_context(flags.FLAGS):
    model_main(flags.FLAGS)

if __name__ == "__main__":
  # define flags
  absl_app.run(main)
```
