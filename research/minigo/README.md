# MiniGo
This is a simplified implementation of MiniGo based on the code provided by the authors: [MiniGo](https://github.com/tensorflow/minigo).

MiniGo is a minimalist Go engine modeled after AlphaGo Zero, built on MuGo. The current implementation consists of three main modules: the DualNet model, the Monte Carlo Tree Search (MCTS), and Go domain knowledge. Currently the **model** part is our focus.

This implementation maintains the features of model training and validation, and also provides evaluation of two Go models.


## DualNet Model
The input to the neural network is a [board_size * board_size * 17] image stack
comprising 17 binary feature planes. 8 feature planes consist of binary values
indicating the presence of the current player's stones; A further 8 feature
planes represent the corresponding features for the opponent's stones; The final
feature plane represents the color to play, and has a constant value of either 1
if black is to play or 0 if white to play. Check `features.py` for more details.

In MiniGo implementation, the input features are processed by a residual tower
that consists of a single convolutional block followed by either 9 or 19
residual blocks.
The convolutional block applies the following modules:
  1. A convolution of num_filter filters of kernel size 3 x 3 with stride 1
  2. Batch normalization
  3. A rectifier non-linearity

Each residual block applies the following modules sequentially to its input:
  1. A convolution of num_filter filters of kernel size 3 x 3 with stride 1
  2. Batch normalization
  3. A rectifier non-linearity
  4. A convolution of num_filter filters of kernel size 3 x 3 with stride 1
  5. Batch normalization
  6. A skip connection that adds the input to the block
  7. A rectifier non-linearity

Note: num_filter is 128 for 19 x 19 board size, and 32 for 9 x 9 board size.

The output of the residual tower is passed into two separate "heads" for
computing the policy and value respectively. The policy head applies the
following modules:
  1. A convolution of 2 filters of kernel size 1 x 1 with stride 1
  2. Batch normalization
  3. A rectifier non-linearity
  4. A fully connected linear layer that outputs a vector of size (board_size * board_size + 1) corresponding to logit probabilities for all intersections and the pass move

The value head applies the following modules:
  1. A convolution of 1 filter of kernel size 1 x 1 with stride 1
  2. Batch normalization
  3. A rectifier non-linearity
  4. A fully connected linear layer to a hidden layer of size 256 for 19 x 19
    board size and 64 for 9x9 board size
  5. A rectifier non-linearity
  6. A fully connected linear layer to a scalar
  7. A tanh non-linearity outputting a scalar in the range [-1, 1]

The overall network depth, in the 10 or 20 block network, is 19 or 39
parameterized layers respectively for the residual tower, plus an additional 2
layers for the policy head and 3 layers for the value head.

## Getting Started
This project assumes you have virtualenv, TensorFlow (>= 1.5) and two other Go-related
packages pygtp(>=0.4) and sgf (==0.5).


## Training Model
One iteration of reinforcement learning consists of the following steps:
 - Bootstrap: initializes a random model
 - Selfplay: plays games with the latest model, producing data used for training
 - Gather: groups games played with the same model into larger files of tfexamples.
 - Train: trains a new model with the selfplay results from the most recent N
   generations.

 Run `minigo.py`.
 ```
 python minigo.py
 ```

## Validating Model
 Run `minigo.py` with `--validation` argument
 ```
 python minigo.py --validation
 ```
 The `--validation` argument is to generate holdout dataset for model validation

## Evaluating MiniGo Models
 Run `minigo.py` with `--evaluation` argument
 ```
 python minigo.py --evaluation
 ```
 The `--evaluation` argument is to invoke the evaluation between the latest model and the current best model.

## Testing Pipeline
As the whole RL pipeline may takes hours to train even for a 9x9 board size, we provide a dummy model with a `--debug` mode for testing purpose.

 Run `minigo.py` with `--debug` argument
 ```
 python minigo.py --debug
 ```
 The `--debug` argument is for testing purpose with a dummy model.

Validation and evaluation can also be tested with the dummy model by combing their corresponding arguments with `--debug`.
To test validation, run the following commands:
 ```
 python minigo.py --debug --validation
 ```
To test evaluation, run the following commands:
 ```
 python minigo.py --debug --evaluation
 ```
To test both validation and evaluation, run the following commands:
 ```
 python minigo.py --debug --validation --evaluation
 ```

## MCTS and Go features (TODO)
Code clean up on MCTS and Go features.
