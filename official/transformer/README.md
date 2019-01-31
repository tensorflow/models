# Transformer Translation Model
This is an implementation of the Transformer translation model as described in the [Attention is All You Need](https://arxiv.org/abs/1706.03762) paper. Based on the code provided by the authors: [Transformer code](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py) from [Tensor2Tensor](https://github.com/tensorflow/tensor2tensor).

Transformer is a neural network architecture that solves sequence to sequence problems using attention mechanisms. Unlike traditional neural seq2seq models, Transformer does not involve recurrent connections. The attention mechanism learns dependencies between tokens in two sequences. Since attention weights apply to all tokens in the sequences, the Transformer model is able to easily capture long-distance dependencies.

Transformer's overall structure follows the standard encoder-decoder pattern. The encoder uses self-attention to compute a representation of the input sequence. The decoder generates the output sequence one token at a time, taking the encoder output and previous decoder-outputted tokens as inputs.

The model also applies embeddings on the input and output tokens, and adds a constant positional encoding. The positional encoding adds information about the position of each token.

## Contents
  * [Contents](#contents)
  * [Walkthrough](#walkthrough)
  * [Benchmarks](#benchmarks)
    * [Training times](#training-times)
    * [Evaluation results](#evaluation-results)
  * [Detailed instructions](#detailed-instructions)
    * [Environment preparation](#environment-preparation)
    * [Download and preprocess datasets](#download-and-preprocess-datasets)
    * [Model training and evaluation](#model-training-and-evaluation)
    * [Translate using the model](#translate-using-the-model)
    * [Compute official BLEU score](#compute-official-bleu-score)
    * [TPU](#tpu)
  * [Export trained model](#export-trained-model)
    * [Example translation](#example-translation)
  * [Implementation overview](#implementation-overview)
    * [Model Definition](#model-definition)
    * [Model Estimator](#model-estimator)
    * [Other scripts](#other-scripts)
    * [Test dataset](#test-dataset)
  * [Term definitions](#term-definitions)

## Walkthrough

Below are the commands for running the Transformer model. See the [Detailed instrutions](#detailed-instructions) for more details on running the model.

```
cd /path/to/models/official/transformer

# Ensure that PYTHONPATH is correctly defined as described in
# https://github.com/tensorflow/models/tree/master/official#requirements
# export PYTHONPATH="$PYTHONPATH:/path/to/models"

# Export variables
PARAM_SET=big
DATA_DIR=$HOME/transformer/data
MODEL_DIR=$HOME/transformer/model_$PARAM_SET
VOCAB_FILE=$DATA_DIR/vocab.ende.32768

# Download training/evaluation datasets
python data_download.py --data_dir=$DATA_DIR

# Train the model for 10 epochs, and evaluate after every epoch.
python transformer_main.py --data_dir=$DATA_DIR --model_dir=$MODEL_DIR \
    --vocab_file=$VOCAB_FILE --param_set=$PARAM_SET \
    --bleu_source=test_data/newstest2014.en --bleu_ref=test_data/newstest2014.de

# Run during training in a separate process to get continuous updates,
# or after training is complete.
tensorboard --logdir=$MODEL_DIR

# Translate some text using the trained model
python translate.py --model_dir=$MODEL_DIR --vocab_file=$VOCAB_FILE \
    --param_set=$PARAM_SET --text="hello world"

# Compute model's BLEU score using the newstest2014 dataset.
python translate.py --model_dir=$MODEL_DIR --vocab_file=$VOCAB_FILE \
    --param_set=$PARAM_SET --file=test_data/newstest2014.en --file_out=translation.en
python compute_bleu.py --translation=translation.en --reference=test_data/newstest2014.de
```

## Benchmarks
### Training times

Currently, both big and base parameter sets run on a single GPU. The measurements below
are reported from running the model on a P100 GPU.

Param Set | batches/sec | batches per epoch | time per epoch
--- | --- | --- | ---
base | 4.8 | 83244 | 4 hr
big | 1.1 | 41365 | 10 hr

### Evaluation results
Below are the case-insensitive BLEU scores after 10 epochs.

Param Set | Score
--- | --- |
base | 27.7
big | 28.9


## Detailed instructions


0. ### Environment preparation

   #### Add models repo to PYTHONPATH
   Follow the instructions described in the [Requirements](https://github.com/tensorflow/models/tree/master/official#requirements) section to add the models folder to the python path.

   #### Export variables (optional)

   Export the following variables, or modify the values in each of the snippets below:
   ```
   PARAM_SET=big
   DATA_DIR=$HOME/transformer/data
   MODEL_DIR=$HOME/transformer/model_$PARAM_SET
   VOCAB_FILE=$DATA_DIR/vocab.ende.32768
   ```

1. ### Download and preprocess datasets

   [data_download.py](data_download.py) downloads and preprocesses the training and evaluation WMT datasets. After the data is downloaded and extracted, the training data is used to generate a vocabulary of subtokens. The evaluation and training strings are tokenized, and the resulting data is sharded, shuffled, and saved as TFRecords.

   1.75GB of compressed data will be downloaded. In total, the raw files (compressed, extracted, and combined files) take up 8.4GB of disk space. The resulting TFRecord and vocabulary files are 722MB. The script takes around 40 minutes to run, with the bulk of the time spent downloading and ~15 minutes spent on preprocessing.

   Command to run:
   ```
   python data_download.py --data_dir=$DATA_DIR
   ```

   Arguments:
   * `--data_dir`: Path where the preprocessed TFRecord data, and vocab file will be saved.
   * Use the `--help` or `-h` flag to get a full list of possible arguments.

2. ### Model training and evaluation

   [transformer_main.py](transformer_main.py) creates a Transformer model, and trains it using Tensorflow Estimator.

   Command to run:
   ```
   python transformer_main.py --data_dir=$DATA_DIR --model_dir=$MODEL_DIR \
       --vocab_file=$VOCAB_FILE --param_set=$PARAM_SET
   ```

   Arguments:
   * `--data_dir`: This should be set to the same directory given to the `data_download`'s `data_dir` argument.
   * `--model_dir`: Directory to save Transformer model training checkpoints.
   * `--vocab_file`: Path to subtoken vacbulary file. If data_download was used, you may find the file in `data_dir`.
   * `--param_set`: Parameter set to use when creating and training the model. Options are `base` and `big` (default).
   * Use the `--help` or `-h` flag to get a full list of possible arguments.

   #### Customizing training schedule

   By default, the model will train for 10 epochs, and evaluate after every epoch. The training schedule may be defined through the flags:
   * Training with epochs (default):
     * `--train_epochs`: The total number of complete passes to make through the dataset
     * `--epochs_between_evals`: The number of epochs to train between evaluations.
   * Training with steps:
     * `--train_steps`: sets the total number of training steps to run.
     * `--steps_between_evals`: Number of training steps to run between evaluations.

   Only one of `train_epochs` or `train_steps` may be set. Since the default option is to evaluate the model after training for an epoch, it may take 4 or more hours between model evaluations. To get more frequent evaluations, use the flags `--train_steps=250000 --steps_between_evals=1000`.

   Note: At the beginning of each training session, the training dataset is reloaded and shuffled. Stopping the training before completing an epoch may result in worse model quality, due to the chance that some examples may be seen more than others. Therefore, it is recommended to use epochs when the model quality is important.

   #### Compute BLEU score during model evaluation

   Use these flags to compute the BLEU when the model evaluates:
   * `--bleu_source`: Path to file containing text to translate.
   * `--bleu_ref`: Path to file containing the reference translation.
   * `--stop_threshold`: Train until the BLEU score reaches this lower bound. This setting overrides the `--train_steps` and `--train_epochs` flags.

   The test source and reference files located in the `test_data` directory are extracted from the preprocessed dataset from the [NMT Seq2Seq tutorial](https://google.github.io/seq2seq/nmt/#download-data).

   When running `transformer_main.py`, use the flags: `--bleu_source=test_data/newstest2014.en --bleu_ref=test_data/newstest2014.de`

   #### Tensorboard
   Training and evaluation metrics (loss, accuracy, approximate BLEU score, etc.) are logged, and can be displayed in the browser using Tensorboard.
   ```
   tensorboard --logdir=$MODEL_DIR
   ```
   The values are displayed at [localhost:6006](localhost:6006).

3. ### Translate using the model
   [translate.py](translate.py) contains the script to use the trained model to translate input text or file. Each line in the file is translated separately.

   Command to run:
   ```
   python translate.py --model_dir=$MODEL_DIR --vocab_file=$VOCAB_FILE \
       --param_set=$PARAM_SET --text="hello world"
   ```

   Arguments for initializing the Subtokenizer and trained model:
   * `--model_dir` and `--param_set`: These parameters are used to rebuild the trained model
   * `--vocab_file`: Path to subtoken vacbulary file. If data_download was used, you may find the file in `data_dir`.

   Arguments for specifying what to translate:
   * `--text`: Text to translate
   * `--file`: Path to file containing text to translate
   * `--file_out`: If `--file` is set, then this file will store the input file's translations.

   To translate the newstest2014 data, run:
   ```
   python translate.py --model_dir=$MODEL_DIR --vocab_file=$VOCAB_FILE \
       --param_set=$PARAM_SET --file=test_data/newstest2014.en --file_out=translation.en
   ```

   Translating the file takes around 15 minutes on a GTX1080, or 5 minutes on a P100.

4. ### Compute official BLEU score
   Use [compute_bleu.py](compute_bleu.py) to compute the BLEU by comparing generated translations to the reference translation.

   Command to run:
   ```
   python compute_bleu.py --translation=translation.en --reference=test_data/newstest2014.de
   ```

   Arguments:
   * `--translation`: Path to file containing generated translations.
   * `--reference`: Path to file containing reference translations.
   * Use the `--help` or `-h` flag to get a full list of possible arguments.
   
5. ### TPU
   TPU support for this version of Transformer is experimental. Currently it is present for
   demonstration purposes only, but will be optimized in the coming weeks.

## Export trained model
To export the model as a Tensorflow [SavedModel](https://www.tensorflow.org/guide/saved_model) format, use the argument `--export_dir` when running `transformer_main.py`. A folder will be created in the directory with the name as the timestamp (e.g. $EXPORT_DIR/1526427396).

```
EXPORT_DIR=$HOME/transformer/saved_model
python transformer_main.py --data_dir=$DATA_DIR --model_dir=$MODEL_DIR \
  --vocab_file=$VOCAB_FILE --param_set=$PARAM_SET --export_model=$EXPORT_DIR
```

To inspect the SavedModel, use saved_model_cli:
```
SAVED_MODEL_DIR=$EXPORT_DIR/{TIMESTAMP}  # replace {TIMESTAMP} with the name of the folder created
saved_model_cli show --dir=$SAVED_MODEL_DIR  --all
```

### Example translation
Let's translate **"hello world!"**, **"goodbye world."**, and **"Would you like some pie?"**.

The SignatureDef for "translate" is:

    signature_def['translate']:
        The given SavedModel SignatureDef contains the following input(s):
          inputs['input'] tensor_info:
              dtype: DT_INT64
              shape: (-1, -1)
              name: Placeholder:0
        The given SavedModel SignatureDef contains the following output(s):
          outputs['outputs'] tensor_info:
              dtype: DT_INT32
              shape: (-1, -1)
              name: model/Transformer/strided_slice_19:0
          outputs['scores'] tensor_info:
              dtype: DT_FLOAT
              shape: (-1)
              name: model/Transformer/strided_slice_20:0

Follow the steps below to use the translate signature def:

1. #### Encode the inputs to integer arrays.
   This can be done using `utils.tokenizer.Subtokenizer`, and the vocab file in the SavedModel assets (`$SAVED_MODEL_DIR/assets.extra/vocab.txt`).

   ```
   from official.transformer.utils.tokenizer import Subtokenizer
   s = Subtokenizer(PATH_TO_VOCAB_FILE)
   print(s.encode("hello world!", add_eos=True))
   ```

   The encoded inputs are:
   * `"hello world!" = [6170, 3731, 178, 207, 1]`
   * `"goodbye world." = [15431, 13966, 36, 178, 3, 1]`
   * `"Would you like some pie?" = [9092, 72, 155, 202, 19851, 102, 1]`

2. #### Run `saved_model_cli` to obtain the predicted translations
   The encoded inputs should be padded so that they are the same length. The padding token is `0`.
   ```
   ENCODED_INPUTS="[[26228, 145, 178, 1, 0, 0, 0], \
                   [15431, 13966, 36, 178, 3, 1, 0], \
                   [9092, 72, 155, 202, 19851, 102, 1]]"
   ```

   Now, use the `run` command with `saved_model_cli` to get the outputs.

   ```
   saved_model_cli run --dir=$SAVED_MODEL_DIR --tag_set=serve --signature_def=translate \
     --input_expr="input=$ENCODED_INPUTS"
   ```

   The outputs will look similar to:
   ```
   Result for output key outputs:
   [[18744   145   297     1     0     0     0     0     0     0     0     0
         0     0]
    [ 5450  4642    21    11   297     3     1     0     0     0     0     0
         0     0]
    [25940    22    66   103 21713    31   102     1     0     0     0     0
         0     0]]
   Result for output key scores:
   [-1.5493642 -1.4032784 -3.252089 ]
   ```

3. #### Decode the outputs to strings.
   Use the `Subtokenizer` and vocab file as described in step 1 to decode the output integer arrays.
   ```
   from official.transformer.utils.tokenizer import Subtokenizer
   s = Subtokenizer(PATH_TO_VOCAB_FILE)
   print(s.decode([18744, 145, 297, 1]))
   ```
   The decoded outputs from above are:
   * `[18744, 145, 297, 1] = "Hallo Welt<EOS>"`
   * `[5450, 4642, 21, 11, 297, 3, 1] = "Abschied von der Welt.<EOS>"`
   * `[25940, 22, 66, 103, 21713, 31, 102, 1] = "Möchten Sie einen Kuchen?<EOS>"`

## Implementation overview

A brief look at each component in the code:

### Model Definition
The [model](model) subdirectory contains the implementation of the Transformer model. The following files define the Transformer model and its layers:
* [transformer.py](model/transformer.py): Defines the transformer model and its encoder/decoder layer stacks.
* [embedding_layer.py](model/embedding_layer.py): Contains the layer that calculates the embeddings. The embedding weights are also used to calculate the pre-softmax probabilities from the decoder output.
* [attention_layer.py](model/attention_layer.py): Defines the multi-headed and self attention layers that are used in the encoder/decoder stacks.
* [ffn_layer.py](model/ffn_layer.py): Defines the feedforward network that is used in the encoder/decoder stacks. The network is composed of 2 fully connected layers.

Other files:
* [beam_search.py](model/beam_search.py) contains the beam search implementation, which is used during model inference to find high scoring translations.
* [model_params.py](model/model_params.py) contains the parameters used for the big and base models.
* [model_utils.py](model/model_utils.py) defines some helper functions used in the model (calculating padding, bias, etc.).


### Model Estimator
[transformer_main.py](model/transformer.py) creates an `Estimator` to train and evaluate the model.

Helper functions:
* [utils/dataset.py](utils/dataset.py): contains functions for creating a `dataset` that is passed to the `Estimator`.
* [utils/metrics.py](utils/metrics.py): defines metrics functions used by the `Estimator` to evaluate the

### Other scripts

Aside from the main file to train the Transformer model, we provide other scripts for using the model or downloading the data:

#### Data download and preprocessing

[data_download.py](data_download.py) downloads and extracts data, then uses `Subtokenizer` to tokenize strings into arrays of int IDs. The int arrays are converted to `tf.Examples` and saved in the `tf.RecordDataset` format.

 The data is downloaded from the Workshop of Machine Transtion (WMT) [news translation task](http://www.statmt.org/wmt17/translation-task.html). The following datasets are used:

 * Europarl v7
 * Common Crawl corpus
 * News Commentary v12

 See the [download section](http://www.statmt.org/wmt17/translation-task.html#download) to explore the raw datasets. The parameters in this model are tuned to fit the English-German translation data, so the EN-DE texts are extracted from the downloaded compressed files.

The text is transformed into arrays of integer IDs using the `Subtokenizer` defined in [`utils/tokenizer.py`](util/tokenizer.py). During initialization of the `Subtokenizer`, the raw training data is used to generate a vocabulary list containing common subtokens.

The target vocabulary size of the WMT dataset is 32,768. The set of subtokens is found through binary search on the minimum number of times a subtoken appears in the data. The actual vocabulary size is 33,708, and is stored in a 324kB file.

#### Translation
Translation is defined in [translate.py](translate.py). First, `Subtokenizer` tokenizes the input. The vocabulary file is the same used to tokenize the training/eval files. Next, beam search is used to find the combination of tokens that maximizes the probability outputted by the model decoder. The tokens are then converted back to strings with `Subtokenizer`.

#### BLEU computation
[compute_bleu.py](compute_bleu.py): Implementation from [https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/bleu_hook.py](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/bleu_hook.py).

### Test dataset
The [newstest2014 files](test_data) are extracted from the [NMT Seq2Seq tutorial](https://google.github.io/seq2seq/nmt/#download-data). The raw text files are converted from the SGM format of the [WMT 2016](http://www.statmt.org/wmt16/translation-task.html) test sets.

## Term definitions

**Steps / Epochs**:
* Step: unit for processing a single batch of data
* Epoch: a complete run through the dataset

Example: Consider a training a dataset with 100 examples that is divided into 20 batches with 5 examples per batch. A single training step trains the model on one batch. After 20 training steps, the model will have trained on every batch in the dataset, or one epoch.

**Subtoken**: Words are referred to as tokens, and parts of words are referred to as 'subtokens'. For example, the word 'inclined' may be split into `['incline', 'd_']`. The '\_' indicates the end of the token. The subtoken vocabulary list is guaranteed to contain the alphabet (including numbers and special characters), so all words can be tokenized.
