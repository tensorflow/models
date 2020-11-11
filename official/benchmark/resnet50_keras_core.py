
def _preprocessing(data):
  return (
      tf.cast(_decode_and_center_crop(data["image"]), tf.float32),
      data["label"],
  )


def _run_benchmark():
  """Runs a resnet50 compile/fit() call and returns the wall time."""
  tmp_dir = tempfile.mkdtemp()
  start_time = time.time()

  batch_size = 64
  dataset = tfds.load(
      "imagenette",
      decoders={"image": tfds.decode.SkipDecoding()},
      split="train",
  )

  dataset = (
      dataset.cache().repeat(
          2
      )  # Artificially increase time per epoch to make it easier to measure
      .map(_preprocessing,
           num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(
               batch_size).prefetch(1))

  with tf.distribute.MirroredStrategy().scope():
    model = tf.keras.applications.ResNet50(weights=None)
    model.compile(
        optimizer=tf.train.experimental.enable_mixed_precision_graph_rewrite(
            tf.keras.optimizers.Adam(), loss_scale="dynamic"),
        loss="sparse_categorical_crossentropy",
    )

  tb_cbk = tf.keras.callbacks.TensorBoard(
      f"{tmp_dir}/{tf.__version__}", profile_batch=300)
  model.fit(dataset, verbose=2, epochs=3, callbacks=[tb_cbk])
  end_time = time.time()
  return end_time - start_time


class Resnet50KerasCoreBenchmark(perfzero_benchmark.PerfZeroBenchmark):

  def benchmark_1_gpu(self):
    wall_time = _run_benchmark()
    self.report_benchmark(iters=-1, wall_time=wall_time)


if __name__ == "__main__":
  tf.test.main()
