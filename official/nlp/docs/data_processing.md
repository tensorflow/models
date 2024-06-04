# TF-NLP Data Processing

## Code locations

Open sourced data processing libraries:
[tensorflow_models/official/nlp/data/](https://github.com/tensorflow/models/tree/28d972a0b30b628cbb7f67a090ea564c3eda99ea/official/nlp/data)

## Preprocess data offline v.s. TFDS

Inside TF-NLP, there are flexible ways to provide training data to the input
pipeline: 1) using python scripts/beam/flume to process/tokenize the data
offline; 2) reading the text data directly from
[TFDS](https://www.tensorflow.org/datasets/api_docs/python/tfds) and using
[TF.Text](https://www.tensorflow.org/tutorials/tensorflow_text/intro) for
tokenization and preprocessing inside the tf.data input pipeline.

### Preprocessing scripts

We have implemented data preprocessing for multiple datasets in the following
python scripts:

*   [create_pretraining_data.py](https://github.com/tensorflow/models/blob/28d972a0b30b628cbb7f67a090ea564c3eda99ea/official/nlp/data/create_pretraining_data.py)

*   [create_finetuning_data.py](https://github.com/tensorflow/models/blob/28d972a0b30b628cbb7f67a090ea564c3eda99ea/official/nlp/data/create_finetuning_data.py)

Then, the processed files with `tf.Example` protos inside should be specified to
the `input_path` argument in
[`DataConfig`](https://github.com/tensorflow/models/blob/28d972a0b30b628cbb7f67a090ea564c3eda99ea/official/core/config_definitions.py#L28).

### TFDS usages

For convenience and consolidation, we built a common
[input_reader.py](https://github.com/tensorflow/models/blob/28d972a0b30b628cbb7f67a090ea564c3eda99ea/official/core/input_reader.py)
library to standardize input reading, which has built-in pass for TFDS.
Specifying the arguments in the
[`DataConfig`](https://github.com/tensorflow/models/blob/28d972a0b30b628cbb7f67a090ea564c3eda99ea/official/core/config_definitions.py#L28),
`tfds_name`, `tfds_data_dir` and `tfds_split`, will let the tf.data pipeline
read from the corresponding dataset inside TFDS.

## DataLoaders

To manage multiple datasets and processing functions, we defined the
[DataLoader](https://github.com/tensorflow/models/blob/28d972a0b30b628cbb7f67a090ea564c3eda99ea/official/nlp/data/data_loader.py)
class to work with the
[data loader factory](https://github.com/tensorflow/models/blob/28d972a0b30b628cbb7f67a090ea564c3eda99ea/official/nlp/data/data_loader_factory.py).

Each dataloader defines the tf.data input pipeline inside the `load` method.

```python
@abc.abstractmethod
def load(
    self,
    input_context: Optional[tf.distribute.InputContext] = None
) -> tf.data.Dataset:
```

Then, the `load` method is called inside each NLP task's `build_input` method
and the trainer wrap that to create distributed datasets.

```python
def build_inputs(self, params, input_context=None):
  """Returns tf.data.Dataset for pretraining."""
  data_loader = YourDataLoader(params)
  return data_loader.load(input_context)
```

By default, in the example above, `params` is the `train_data` or
`validation_data` field of the `task` field of the experiment config. `params`
is a type of `DataConfig`.

It is important to note that, for TPU training, the entire `load` method will
run on the TPU workers and it requires that the function does not access
resources outside, e.g. the task attributes.

To work with raw text features, we need to use the `DataLoader`s handling the
text data with TF.Text. You can take the following dataloaders as references:

*   [sentence_prediction_dataloader.py](https://github.com/tensorflow/models/blob/28d972a0b30b628cbb7f67a090ea564c3eda99ea/official/nlp/data/sentence_prediction_dataloader.py)
    for BERT GLUE fine tuning using TFDS with raw text features.

## Speed up training using TF.data service and dynamic sequence length on TPUs

With TF 2.x, we can enable some types of dynamic shapes on TPUs, thanks to TF
2.x programing model and TPUStrategy/XLA works.

Depending on the data distribution, we are seeing 50% to 90% speed up on typical
text data for BERT pretraining applications relative to padded static shape
inputs.

To enable dynamic sequence, we need to use
`tf data service` for the global bucketizing over
sequences. To enable it, you can simply add `--enable_tf_data_service` when you
start experiments.

To pair with tf data service, we need to use the dataloaders that has the
bucketizing function implemented. You can take the following dataloaders as
references:

*   [pretrain_dynamic_dataloader.py](https://github.com/tensorflow/models/blob/28d972a0b30b628cbb7f67a090ea564c3eda99ea/official/nlp/data/pretrain_dynamic_dataloader.py)
    for BERT pretraining on the tokenized datasets.
