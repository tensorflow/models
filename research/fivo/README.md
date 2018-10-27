# Filtering Variational Objectives

This folder contains a TensorFlow implementation of the algorithms from

Chris J. Maddison\*, Dieterich Lawson\*, George Tucker\*, Nicolas Heess, Mohammad Norouzi, Andriy Mnih, Arnaud Doucet, and Yee Whye Teh. "Filtering Variational Objectives." NIPS 2017.

[https://arxiv.org/abs/1705.09279](https://arxiv.org/abs/1705.09279)

This code implements 3 different bounds for training sequential latent variable models: the evidence lower bound (ELBO), the importance weighted auto-encoder bound (IWAE), and our bound, the filtering variational objective (FIVO).

Additionally it contains several sequential latent variable model implementations:

* Variational recurrent neural network (VRNN)
* Stochastic recurrent neural network (SRNN)
* Gaussian hidden Markov model with linear conditionals (GHMM)

The VRNN and SRNN can be trained for sequence modeling of pianoroll and speech data. The GHMM is trainable on a synthetic dataset, useful as a simple example of an analytically tractable model.

#### Directory Structure
The important parts of the code are organized as follows.

```
run_fivo.py         # main script, contains flag definitions
fivo
├─smc.py            # a sequential Monte Carlo implementation
├─bounds.py         # code for computing each bound, uses smc.py
├─runners.py        # code for VRNN and SRNN training and evaluation
├─ghmm_runners.py   # code for GHMM training and evaluation
├─data
| ├─datasets.py     # readers for pianoroll and speech datasets
| ├─calculate_pianoroll_mean.py  # preprocesses the pianoroll datasets
| └─create_timit_dataset.py      # preprocesses the TIMIT dataset
└─models
  ├─base.py   # base classes used in other models
  ├─vrnn.py   # VRNN implementation
  ├─srnn.py   # SRNN implementation
  └─ghmm.py   # Gaussian hidden Markov model (GHMM) implementation
bin
├─run_train.sh            # an example script that runs training
├─run_eval.sh             # an example script that runs evaluation
├─run_sample.sh           # an example script that runs sampling
├─run_tests.sh            # a script that runs all tests
└─download_pianorolls.sh  # a script that downloads pianoroll files
```

### Pianorolls

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

Now we can train a model. Here is the command for a standard training run, taken from `bin/run_train.sh`:
```
python run_fivo.py \
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

You should see output that looks something like this (with extra logging cruft):

```
Saving checkpoints for 0 into /tmp/fivo/model.ckpt.
Step 1, fivo bound per timestep: -11.322491
global_step/sec: 7.49971
Step 101, fivo bound per timestep: -11.399275
global_step/sec: 8.04498
Step 201, fivo bound per timestep: -11.174991
global_step/sec: 8.03989
Step 301, fivo bound per timestep: -11.073008
```
#### Evaluation

You can also evaluate saved checkpoints. The `eval` mode loads a model checkpoint, tests its performance on all items in a dataset, and reports the log-likelihood averaged over the dataset. For example here is a command, taken from `bin/run_eval.sh`, that will evaluate a JSB model on the test set:

```
python run_fivo.py \
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
Restoring parameters from /tmp/fivo/model.ckpt-0
Model restored from step 0, evaluating.
test elbo ll/t: -12.198834, iwae ll/t: -11.981187 fivo ll/t: -11.579776
test elbo ll/seq: -748.564789, iwae ll/seq: -735.209206 fivo ll/seq: -710.577141
```
The evaluation script prints log-likelihood in both nats per timestep (ll/t) and nats per sequence (ll/seq) for all three bounds.

#### Sampling

You can also sample from trained models. The `sample` mode loads a model checkpoint, conditions the model on a prefix of a randomly chosen datapoint, samples a sequence of outputs from the conditioned model, and writes out the samples and prefix to a `.npz` file in `logdir`. For example here is a command that samples from a model trained on JSB, taken from `bin/run_sample.sh`:
```
python run_fivo.py \
  --mode=sample \
  --alsologtostderr \
  --logdir="/tmp/fivo" \
  --model=vrnn \
  --bound=fivo \
  --batch_size=4 \
  --num_samples=4 \
  --split=test \
  --dataset_path="$PIANOROLL_DIR/jsb.pkl" \
  --dataset_type="pianoroll" \
  --prefix_length=25 \
  --sample_length=50
```

Here `num_samples` denotes the number of samples used when conditioning the model as well as the number of trajectories to sample for each prefix.

You should see very little output.
```
Restoring parameters from /tmp/fivo/model.ckpt-0
Running local_init_op.
Done running local_init_op.
```

Loading the samples with `np.load` confirms that we conditioned the model on 4
prefixes of length 25 and sampled 4 sequences of length 50 for each prefix.
```
>>> import numpy as np
>>> x = np.load("/tmp/fivo/samples.npz")
>>> x[()]['prefixes'].shape
(25, 4, 88)
>>> x[()]['samples'].shape
(50, 4, 4, 88)
```

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
python run_fivo.py \
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
Evaluation and sampling are similar.

### Tests
This codebase comes with a number of tests to verify correctness, runnable via `bin/run_tests.sh`. The tests are also useful to look at for examples of how to use the code.

### Contact

This codebase is maintained by Dieterich Lawson, reachable via email at dieterichl@google.com. For questions and issues please open an issue on the tensorflow/models issues tracker and assign it to @dieterichlawson.
