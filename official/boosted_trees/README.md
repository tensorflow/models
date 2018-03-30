# Classifying Higgs boson processes in the HIGGS Data Set
## Overview
The [HIGGS Data Set](https://archive.ics.uci.edu/ml/datasets/HIGGS) contains 11 million samples with 28 features, and is for the classification problem to distinguish between a signal process which produces Higgs bosons and a background process which does not.

We use Gradient Boosted Trees algorithm to distinguish the two classes.

---

The code sample in this directory uses the high level `tf.estimator.Estimator` API. This API is great for fast iteration and quickly adapting models to your own datasets without major code overhauls. It allows you to move from single-worker training to distributed training, and it makes it easy to export model binaries for prediction. Here, for further simplicity and faster execution, we use a utility function `tf.contrib.estimator.boosted_trees_classifier_train_in_memory`. This utility function is especially effective when the input is provided as in-memory data sets like numpy arrays.

An input function for the `Estimator` typically uses `tf.data.Dataset` API, which can handle various data control like streaming, batching, transform and shuffling. However `boosted_trees_classifier_train_in_memory()` utility function requires that the entire data is provided as a single batch (i.e. without using `batch()` API). Thus in this practice, simply `Dataset.from_tensors()` is used to convert numpy arrays into structured tensors, and `Dataset.zip()` is used to put features and label together.
For further references of `Dataset`, [Read more here](https://www.tensorflow.org/programmers_guide/datasets).

## Running the code
First make sure you've [added the models folder to your Python path](/official/#running-the-models); otherwise you may encounter an error like `ImportError: No module named official.boosted_trees`.

### Setup
The [HIGGS Data Set](https://archive.ics.uci.edu/ml/datasets/HIGGS) that this sample uses for training is hosted by the [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/). We have provided a script that downloads and cleans the necessary files.

```
python data_download.py
```

This will download a file and store the processed file under the directory designated by `--data_dir` (defaults to `/tmp/higgs_data/`). To change the target directory, set the `--data_dir` flag. The directory could be network storages that Tensorflow supports (like Google Cloud Storage, `gs://<bucket>/<path>/`).
The file downloaded to the local temporary folder is about 2.8 GB, and the processed file is about 1.3G, so there should be enough storage to handle them.


### Training
You can run the code locally as follows:

```
python train_higgs.py
```

The model is saved to `/tmp/higgs_model` by default, which can be changed using the `--model_dir` flag. To train a new model after one, provide a new `--model_dir` value, or remove the dir set by `--model_dir` before launching.

Model parameters can be adjusted by flags, like `--n_trees`, `--max_depth`, `--learning_rate`.

The final accuacy will be around 74% and loss will be around 0.5165 over the eval set, when trained with the default parameters.

By default, the first 1 million examples among 11 millions are used for training, and the last 1 million examples are used for evaluation.
The training/evaluation data can be selected as index ranges by flags `--train_start`, `--train_count`, `--eval_start`, `--eval_count`, etc.

### TensorBoard

Run TensorBoard to inspect the details about the graph and training progression.

```
tensorboard --logdir=/tmp/higgs_model
```

## Additional Links

If you are interested in distributed training, take a look at [Distributed TensorFlow](https://www.tensorflow.org/deploy/distributed).

You can also [train models on Cloud ML Engine](https://cloud.google.com/ml-engine/docs/getting-started-training-prediction), which provides [hyperparameter tuning](https://cloud.google.com/ml-engine/docs/getting-started-training-prediction#hyperparameter_tuning) to maximize your model's results and enables [deploying your model for prediction](https://cloud.google.com/ml-engine/docs/getting-started-training-prediction#deploy_a_model_to_support_prediction).
