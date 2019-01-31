# Classifying Higgs boson processes in the HIGGS Data Set
## Overview
The [HIGGS Data Set](https://archive.ics.uci.edu/ml/datasets/HIGGS) contains 11 million samples with 28 features, and is for the classification problem to distinguish between a signal process which produces Higgs bosons and a background process which does not.

We use Gradient Boosted Trees algorithm to distinguish the two classes.

---

The code sample uses the high level `tf.estimator.Estimator` and `tf.data.Dataset`.  These APIs are great for fast iteration and quickly adapting models to your own datasets without major code overhauls.  It allows you to move from single-worker training to distributed training, and makes it easy to export model binaries for prediction.  Here, for further simplicity and faster execution, we use a utility function `tf.contrib.estimator.boosted_trees_classifier_train_in_memory`.  This utility function is especially effective when the input is provided as in-memory data sets like numpy arrays.

An input function for the `Estimator` typically uses `tf.data.Dataset` API, which can handle various data control like streaming, batching, transform and shuffling. However `boosted_trees_classifier_train_in_memory()` utility function requires that the entire data is provided as a single batch (i.e. without using `batch()` API). Thus in this practice, simply `Dataset.from_tensors()` is used to convert numpy arrays into structured tensors, and `Dataset.zip()` is used to put features and label together.
For further references of `Dataset`, [Read more here](https://www.tensorflow.org/guide/datasets).

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

The final accuracy will be around 74% and loss will be around 0.516 over the eval set, when trained with the default parameters.

By default, the first 1 million examples among 11 millions are used for training, and the last 1 million examples are used for evaluation.
The training/evaluation data can be selected as index ranges by flags `--train_start`, `--train_count`, `--eval_start`, `--eval_count`, etc.

### TensorBoard

Run TensorBoard to inspect the details about the graph and training progression.

```
tensorboard --logdir=/tmp/higgs_model  # set logdir as --model_dir set during training.
```

## Inference with SavedModel
You can export the model into Tensorflow [SavedModel](https://www.tensorflow.org/guide/saved_model) format by using the argument `--export_dir`:

```
python train_higgs.py --export_dir /tmp/higgs_boosted_trees_saved_model
```

After the model finishes training, use [`saved_model_cli`](https://www.tensorflow.org/guide/saved_model#cli_to_inspect_and_execute_savedmodel) to inspect and execute the SavedModel.

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
Let's use the model to predict the income group of two examples.
Note that this model exports SavedModel with the custom parsing module that accepts csv lines as features. (Each line is an example with 28 columns; be careful to not add a label column, unlike in the training data.)

```
saved_model_cli run --dir /tmp/boosted_trees_higgs_saved_model/${TIMESTAMP}/ \
    --tag_set serve --signature_def="predict" \
    --input_exprs='inputs=["0.869293,-0.635082,0.225690,0.327470,-0.689993,0.754202,-0.248573,-1.092064,0.0,1.374992,-0.653674,0.930349,1.107436,1.138904,-1.578198,-1.046985,0.0,0.657930,-0.010455,-0.045767,3.101961,1.353760,0.979563,0.978076,0.920005,0.721657,0.988751,0.876678", "1.595839,-0.607811,0.007075,1.818450,-0.111906,0.847550,-0.566437,1.581239,2.173076,0.755421,0.643110,1.426367,0.0,0.921661,-1.190432,-1.615589,0.0,0.651114,-0.654227,-1.274345,3.101961,0.823761,0.938191,0.971758,0.789176,0.430553,0.961357,0.957818"]'
```

This will print out the predicted classes and class probabilities. Something like:

```
Result for output key class_ids:
[[1]
 [0]]
Result for output key classes:
[['1']
 ['0']]
Result for output key logistic:
[[0.6440273 ]
 [0.10902369]]
Result for output key logits:
[[ 0.59288704]
 [-2.1007526 ]]
Result for output key probabilities:
[[0.3559727 0.6440273]
 [0.8909763 0.1090237]]
```

Please note that "predict" signature_def gives out different (more detailed) results than "classification" or "serving_default".

## Additional Links

If you are interested in distributed training, take a look at [Distributed TensorFlow](https://www.tensorflow.org/deploy/distributed).

You can also [train models on Cloud ML Engine](https://cloud.google.com/ml-engine/docs/getting-started-training-prediction), which provides [hyperparameter tuning](https://cloud.google.com/ml-engine/docs/getting-started-training-prediction#hyperparameter_tuning) to maximize your model's results and enables [deploying your model for prediction](https://cloud.google.com/ml-engine/docs/getting-started-training-prediction#deploy_a_model_to_support_prediction).
