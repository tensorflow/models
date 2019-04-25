from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import sklearn.preprocessing as prep
import tensorflow as tf
import tensorflow.keras.layers as layers

from autoencoder_models.Autoencoder import Autoencoder

mnist = tf.keras.datasets.mnist


def standard_scale(X_train, X_test):
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_train, X_test


def get_random_block_from_data(data, batch_size):
    start_index = np.random.randint(0, len(data) - batch_size)
    return data[start_index:(start_index + batch_size)]


(X_train, _), (X_test, _) = mnist.load_data()
X_train = tf.cast(np.reshape(X_train, (X_train.shape[0], X_train.shape[1] * X_train.shape[2])), tf.float64)
X_test = tf.cast(np.reshape(X_test, (X_test.shape[0], X_test.shape[1] * X_test.shape[2])), tf.float64)

X_train, X_test = standard_scale(X_train, X_test)

train_data = tf.data.Dataset.from_tensor_slices(X_train).batch(128).shuffle(buffer_size=1024)
test_data = tf.data.Dataset.from_tensor_slices(X_test).batch(128).shuffle(buffer_size=512)

n_samples = int(len(X_train) + len(X_test))
training_epochs = 20
batch_size = 128
display_step = 1

optimizer = tf.optimizers.Adam(learning_rate=0.01)
mse_loss = tf.keras.losses.MeanSquaredError()
loss_metric = tf.keras.metrics.Mean()

autoencoder = Autoencoder([200, 394, 784])

# Iterate over epochs.
for epoch in range(10):
    print(f'Epoch {epoch+1}')

  # Iterate over the batches of the dataset.
    for step, x_batch in enumerate(train_data):
        with tf.GradientTape() as tape:
          recon = autoencoder(x_batch)
          loss = mse_loss(x_batch, recon)

        grads = tape.gradient(loss, autoencoder.trainable_variables)
        optimizer.apply_gradients(zip(grads, autoencoder.trainable_variables))

        loss_metric(loss)

        if step % 100 == 0:
          print(f'Step {step}: mean loss = {loss_metric.result()}')