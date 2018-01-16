# PTB Language Model
This example code trains a [language model](https://en.wikipedia.org/wiki/Language_model) on the [Penn TreeBank](https://catalog.ldc.upenn.edu/ldc99t42) dataset ([available for download here](http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz)).

The model is implemented based on [*Recurrent Neural Network Regularization* (Zaremba et. al, 2014)](https://arxiv.org/abs/1409.2329). It incorporates 2 layers of LSTM cells with dropout applied as described in the paper.

## Running the model

1. **Download and extract the PTB dataset: http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz**

   The training, validation, and testing files are located in the `data` folder.

2. **Train the model with [`run_training.py`](run_training.py):**
   ```
   python run_training.py --data_dir={DATA PATH}
   ```
   Required arguments:
   * `--data_dir`: path to the training/validation/testing files. (e.g. `.../Downloads/simple-examples/data`)

   Other arguments:
   * `--model`: The model hyperparameters/configuration to use.
         Options are: `small`, `medium` (default), and `large`.
   * `--model_dir`: directory where the model will be saved. The model will be saved to `/tmp/rnn_ptb_model` by default.
   * `--reset_training`: Whether to clear the model directory before training (default `false`).

   **Visualizing training progress with TensorBoard**

   At any point during/after the training, you can see the current training and validation perplexity using TensorBoard:
   ```
   tensorboard --logdir={MODEL DIR}
   ```
3. **Generate text with [`run_predict.py`](run_predict.py)**:
   ```
   python run_predict.py --data_dir={DATA PATH}
   ```

   The arguments `--data_dir` (required), `--model`, and `--model_dir` should be set to the same values as in the previous step.

   Prediction arguments:
   * `--input`: Starting string that will be used to warm up the state of the RNN.
   * `--num_predictions`: Number of words to generate.

### Tests and benchmarks
[`ptb_test.py`](ptb_test.py) defines tests and benchmarks for the model.
* The test class `TestModel` ensures that the outputs from evaluation and prediction have the correct shapes. In addition, it trains the model for 20 steps on dummy data to test that the loss decreases as expected.
* The `Benchmarks` class times the average number of examples that are trained per second, averaged over 100 or 1000 iterations (100 iterations for cpu, 1000 for gpu).

Use the commands below to run just the tests, or both tests and benchmarks:
```
python ptb_test.py
```
```
python ptb_test.py --benchmarks=all
```

## Results
 model|# epochs|train perplexity|valid perplexity|test perplexity
 --- | --- | --- | --- | ---
 small|13|35.31|119.70|114.67
 medium|39|50.80|86.35|83.81
 large|55|41.98|82.24|78.12

Training the `medium` model for 39 epochs takes about an hour on a single GTX 1080 GPU.
The `large` model takes around 4 hours to train for 55 epochs.

## Implementation details
*File Overview*

* [`model.py`](model.py): Defines the PTB model.
* [`model_params.py`](model_params.py): Defines various hyperparameters for the model (batch size, learning rate, etc.)
* [`run_training.py`](run_training.py): Constructs and trains the `Estimator` built around the PTB model.
* [`run_predict.py`](run_predict.py): Example illustrating how to use the model to generate text.
* [`util.py`](util.py): Contains utility functions

### Model
*Defined in [`model.py`](model.py)*

The `PTBModel` class handles the following:
* constructs the model
* updates the internel state of the LSTM cells
* calculates the expected logits when given an input batch

The model consists of the word embedding layer, rnn layers, and softmax layers. If the model is constructed during the training phase, dropout layers are also added to each rnn layer.

### Model training and evaluation
*Defined in [`run_training.py`](run_training.py)*


Constructs a `tf.estimator.Estimator` to train and evaluate the model. The estimator is defined by an input function and a model function.

#### `input_fn`
Provides features and labels (if training or evaluating) to the `model_fn`.

The input data is batched so that adjacent batches sequentially traverse the data. This is illustrated in the example below.

* **Batching example**

   Assume that the dataset is a sequential list of numbers from 1 to 24: `{1, 2, 3, ..., 24}`

   The dataset is batched with the following parameters:
   * `batch_size = 2`: Number of examples per batch
   * `unrolled_steps = 3`: The number of times the RNN is unrolled, i.e., the number of words per example.

   The batched data will look like:

   Batch 1| Batch 2 | Batch 3 | Batch 4
   ---|---|---|---
   1, 2, 3 | 4, 5, 6 | 7, 8, 9 | 10, 11, 12 
   13, 14, 15 | 16, 17, 18 | 19, 20, 21|22, 23, 24

   The first training iteration will receive the input:
   ```
   1, 2, 3
   13, 14, 15
   ```
   where the examples are `1, 2, 3` and `13, 14, 15`.

   The next iteration will receive the examples `4, 5, 6` and `16, 17, 18`.

This batching technique allows the hidden state from the previous batch to carry over.


#### `model_fn`
Constructs the model. Uses the features and labels provided by the `input_fn` to train the variables in the model, and calculate the loss.

An instance of `PTBModel` is constructed in the `model_fn`. The `PTBModel` returns the expected logits, which are used to calculate the loss and gradient for training.

