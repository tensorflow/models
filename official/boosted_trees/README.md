# Classifying Higgs boson processes in the HIGGS Data Set
## Overview
The [HIGGS Data Set](https://archive.ics.uci.edu/ml/datasets/HIGGS) contains 11 million samples with 28 features, and is for the classification problem to distinguish between a signal process which produces Higgs bosons and a background process which does not.

We use Gradient Boosted Trees algorithm to distinguish the two classes.

---

The code sample uses the high level `tf.estimator.Estimator` and `tf.data.Dataset`.  These APIs are great for fast iteration and quickly adapting models to your own datasets without major code overhauls.  It allows you to move from single-worker training to distributed training, and makes it easy to export model binaries for prediction.  Here, for further simplicity and faster execution, we use a utility function `tf.contrib.estimator.boosted_trees_classifier_train_in_memory`.  This utility function is especially effective when the input is provided as in-memory data sets like numpy arrays.

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
The file downloaded to the local temporary folder is about 2.8 GB, and the processed file is about 0.8 GB, so there should be enough storage to handle them.


### Training

This example uses about 3 GB of RAM during training.
You can run the code locally as follows:

```
python train_higgs.py
```

The model is by default saved to `/tmp/higgs_model`, which can be changed using the `--model_dir` flag.
Note that the model_dir is cleaned up before every time training starts.

Model parameters can be adjusted by flags, like `--n_trees`, `--max_depth`, `--learning_rate` and so on.  Check out the code for details.

The final accuacy will be around 74% and loss will be around 0.516 over the eval set, when trained with the default parameters.

By default, the first 1 million examples among 11 millions are used for training, and the last 1 million examples are used for evaluation.
The training/evaluation data can be selected as index ranges by flags `--train_start`, `--train_count`, `--eval_start`, `--eval_count`, etc.

### TensorBoard

Run TensorBoard to inspect the details about the graph and training progression.

```
tensorboard --logdir=/tmp/higgs_model  # set logdir as --model_dir set during training.
```

## Inference with SavedModel
You can export the model into Tensorflow [SavedModel](https://www.tensorflow.org/programmers_guide/saved_model) format by using the argument `--export_dir`:

```
python train_higgs.py --export_dir /tmp/higgs_boosted_trees_saved_model
```

After the model finishes training, use [`saved_model_cli`](https://www.tensorflow.org/programmers_guide/saved_model#cli_to_inspect_and_execute_savedmodel) to inspect and execute the SavedModel.

Try the following commands to inspect the SavedModel:

**Replace `${TIMESTAMP}` with the folder produced (e.g. 1524249124)**
```
# List possible tag_sets. Only one metagraph is saved, so there will be one option.
saved_model_cli show --dir /tmp/higgs_boosted_trees_saved_model/${TIMESTAMP}/

# Show SignatureDefs for tag_set=serve. SignatureDefs define the outputs to show.
saved_model_cli show --dir /tmp/higgs_boosted_trees_saved_model/${TIMESTAMP}/ \
    --tag_set serve --all
```

### Inference
Let's use the model to predict the income group of two examples:

```
saved_model_cli run --dir /tmp/boosted_trees_higgs_saved_model/${TIMESTAMP}/ \
    --tag_set serve --signature_def="predict" \
    --input_examples='examples=[{"feature_01":[0.8692932],"feature_02":[-0.6350818],"feature_03":[0.2256903],"feature_04":[0.3274701],"feature_05":[-0.6899932],"feature_06":[0.7542022],"feature_07":[-0.2485731],"feature_08":[-1.0920639],"feature_09":[0.0],"feature_10":[1.3749921],"feature_11":[-0.6536742],"feature_12":[0.9303491],"feature_13":[1.1074361],"feature_14":[1.1389043],"feature_15":[-1.5781983],"feature_16":[-1.0469854],"feature_17":[0.0],"feature_18":[0.6579295],"feature_19":[-0.0104546],"feature_20":[-0.0457672],"feature_21":[3.1019614],"feature_22":[1.3537600],"feature_23":[0.9795631],"feature_24":[0.9780762],"feature_25":[0.9200048],"feature_26":[0.7216575],"feature_27":[0.9887509],"feature_28":[0.8766783]}, {"feature_01":[1.5958393],"feature_02":[-0.6078107],"feature_03":[0.0070749],"feature_04":[1.8184496],"feature_05":[-0.1119060],"feature_06":[0.8475499],"feature_07":[-0.5664370],"feature_08":[1.5812393],"feature_09":[2.1730762],"feature_10":[0.7554210],"feature_11":[0.6431096],"feature_12":[1.4263668],"feature_13":[0.0],"feature_14":[0.9216608],"feature_15":[-1.1904324],"feature_16":[-1.6155890],"feature_17":[0.0],"feature_18":[0.6511141],"feature_19":[-0.6542270],"feature_20":[-1.2743449],"feature_21":[3.1019614],"feature_22":[0.8237606],"feature_23":[0.9381914],"feature_24":[0.9717582],"feature_25":[0.7891763],"feature_26":[0.4305533],"feature_27":[0.9613569],"feature_28":[0.9578179]}]'
```

This will print out the predicted classes and class probabilities.

## Additional Links

If you are interested in distributed training, take a look at [Distributed TensorFlow](https://www.tensorflow.org/deploy/distributed).

You can also [train models on Cloud ML Engine](https://cloud.google.com/ml-engine/docs/getting-started-training-prediction), which provides [hyperparameter tuning](https://cloud.google.com/ml-engine/docs/getting-started-training-prediction#hyperparameter_tuning) to maximize your model's results and enables [deploying your model for prediction](https://cloud.google.com/ml-engine/docs/getting-started-training-prediction#deploy_a_model_to_support_prediction).
