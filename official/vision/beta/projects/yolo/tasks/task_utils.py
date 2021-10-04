import tensorflow as tf


class ListMetrics:
  """Private class used to cleanly place the matric values for each level."""

  def __init__(self, metric_names, name="ListMetrics", **kwargs):
    self.name = name
    self._metric_names = metric_names
    self._metrics = self.build_metric()
    return

  def build_metric(self):
    metric_names = self._metric_names
    metrics = []
    for name in metric_names:
      metrics.append(tf.keras.metrics.Mean(name, dtype=tf.float32))
    return metrics

  def update_state(self, loss_metrics):
    metrics = self._metrics
    for m in metrics:
      m.update_state(loss_metrics[m.name])
    return

  def result(self):
    logs = dict()
    metrics = self._metrics
    for m in metrics:
      logs.update({m.name: m.result()})
    return logs

  def reset_states(self):
    metrics = self._metrics
    for m in metrics:
      m.reset_states()
    return