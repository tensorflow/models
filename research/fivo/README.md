# Filtering Variational Objectives

This folder contains a TensorFlow implementation of the algorithms from

Chris J. Maddison\*, Dieterich Lawson\*, George Tucker\*, Nicolas Heess, Mohammad Norouzi, Andriy Mnih, Arnaud Doucet, and Yee Whye Teh. "Filtering Variational Objectives." NIPS 2017.

[https://arxiv.org/abs/1705.09279](https://arxiv.org/abs/1705.09279)

This code implements 3 different bounds for training sequential latent variable models: the evidence lower bound (ELBO), the importance weighted auto-encoder bound (IWAE), and our bound, the filtering variational objective (FIVO).

Additionally it contains an implementation of the variational recurrent neural network (VRNN), a sequential latent variable model that can be trained using these three objectives. This repo provides code for training a VRNN to do sequence modeling of pianoroll and speech data.

#### Directory Structure
The important parts of the code are organized as follows.

```
fivo.py           # main script, contains flag definitions
runners.py        # graph construction code for training and evaluation
bounds.py         # code for computing each bound
data
├── datasets.py                    # readers for pianoroll and speech datasets
├── calculate_pianoroll_mean.py    # preprocesses the pianoroll datasets
└── create_timit_dataset.py        # preprocesses the TIMIT dataset
models
└── vrnn.py       # variational RNN implementation
bin
├── run_train.sh            # an example script that runs training
├── run_eval.sh             # an example script that runs evaluation
└── download_pianorolls.sh  # a script that downloads the pianoroll files
```

### Training on Pianorolls

Requirements before we start:

* TensorFlow (see [tensorflow.org](http://tensorflow.org) for how to install)
* [scipy](https://www.scipy.org/)
* [sonnet](https://github.com/deepmind/sonnet)


#### Download the Data

The pianoroll datasets are encoded as pickled sparse arrays and are available at [http://www-etud.iro.umontreal.ca/~boulanni/icml2012](http://www-etud.iro.umontreal.ca/~boulanni/icml2012). You can use the script `bin/download_pianorolls.sh` to download the files into a directory of your choosing.
```
export PIANOROLL_DIR=~/pianorolls
mkdir $PIANOROLL_DIR
sh bin/download_pianorolls.sh $PIANOROLL_DIR
```

#### Preprocess the Data

The script `calculate_pianoroll_mean.py` loads a pianoroll pickle file, calculates the mean, updates the pickle file to include the mean under the key `train_mean`, and writes the file back to disk in-place. You should do this for all pianoroll datasets you wish to train on.

```
python data/calculate_pianoroll_mean.py --in_file=$PIANOROLL_DIR/piano-midi.de.pkl
python data/calculate_pianoroll_mean.py --in_file=$PIANOROLL_DIR/nottingham.de.pkl
python data/calculate_pianoroll_mean.py --in_file=$PIANOROLL_DIR/musedata.pkl
python data/calculate_pianoroll_mean.py --in_file=$PIANOROLL_DIR/jsb.pkl
```

#### Training

Now we can train a model. Here is a standard training run, taken from `bin/run_train.sh`:
```
python fivo.py \
  --mode=train \
  --logdir=/tmp/fivo \
  --model=vrnn \
  --bound=fivo \
  --summarize_every=100 \
  --batch_size=4 \
  --num_samples=4 \
  --learning_rate=0.0001 \
  --dataset_path="$PIANOROLL_DIR/jsb.pkl" \
  --dataset_type="pianoroll"
```

You should see output that looks something like this (with a lot of extra logging cruft):

```
Step 1, fivo bound per timestep: -11.801050
global_step/sec: 9.89825
Step 101, fivo bound per timestep: -11.198309
global_step/sec: 9.55475
Step 201, fivo bound per timestep: -11.287262
global_step/sec: 9.68146
step 301, fivo bound per timestep: -11.316490
global_step/sec: 9.94295
Step 401, fivo bound per timestep: -11.151743
```
You will also see lines saying `Out of range: exceptions.StopIteration: Iteration finished`. This is not an error and is fine.
#### Evaluation

You can also evaluate saved checkpoints. The `eval` mode loads a model checkpoint, tests its performance on all items in a dataset, and reports the log-likelihood averaged over the dataset. For example here is a command, taken from `bin/run_eval.sh`, that will evaluate a JSB model on the test set:

```
python fivo.py \
  --mode=eval \
  --split=test \
  --alsologtostderr \
  --logdir=/tmp/fivo \
  --model=vrnn \
  --batch_size=4 \
  --num_samples=4 \
  --dataset_path="$PIANOROLL_DIR/jsb.pkl" \
  --dataset_type="pianoroll"
```

You should see output like this:
```
Model restored from step 1, evaluating.
test elbo ll/t: -12.299635, iwae ll/t: -12.128336 fivo ll/t: -11.656939
test elbo ll/seq: -754.750312, iwae ll/seq: -744.238773 fivo ll/seq: -715.3121490
```
The evaluation script prints log-likelihood in both nats per timestep (ll/t) and nats per sequence (ll/seq) for all three bounds.

### Training on TIMIT

The TIMIT speech dataset is available at the [Linguistic Data Consortium website](https://catalog.ldc.upenn.edu/LDC93S1), but is unfortunately not free. These instructions will proceed assuming you have downloaded the TIMIT archive and extracted it into the directory `$RAW_TIMIT_DIR`.

#### Preprocess TIMIT

We preprocess TIMIT (as described in our paper) and write it out to a series of TFRecord files. To prepare the TIMIT dataset use the script `create_timit_dataset.py`
```
export $TIMIT_DIR=~/timit_dataset
mkdir $TIMIT_DIR
python data/create_timit_dataset.py \
  --raw_timit_dir=$RAW_TIMIT_DIR \
  --out_dir=$TIMIT_DIR
```
You should see this exact output:
```
4389 train / 231 valid / 1680 test
train mean: 0.006060  train std: 548.136169
```

#### Training on TIMIT
This is very similar to training on pianoroll datasets, with just a few flags switched.
```
python fivo.py \
  --mode=train \
  --logdir=/tmp/fivo \
  --model=vrnn \
  --bound=fivo \
  --summarize_every=100 \
  --batch_size=4 \
  --num_samples=4 \
  --learning_rate=0.0001 \
  --dataset_path="$TIMIT_DIR/train" \
  --dataset_type="speech"
```

### Contact

This codebase is maintained by Dieterich Lawson, reachable via email at dieterichl@google.com. For questions and issues please open an issue on the tensorflow/models issues tracker and assign it to @dieterichlawson.
