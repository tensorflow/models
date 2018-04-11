# MiniGo
This is a simplified implementation of MiniGo based on the code provided by the authors: [MiniGo](https://github.com/tensorflow/minigo).

MiniGo is a minimalist Go engine modeled after AlphaGo Zero, built on MuGo. The current implementation consists of three main modules: the DaulNet model, the Monte Carlo Tree Search (MCTS), and Go domain knowledge. Currently the **model** part is our focus.

This implementation maintains the features of model training and validation, and also provides evaluation of two Go models.

## Getting Started
Please follow the [instructions](https://github.com/tensorflow/minigo/blob/master/README.md#getting-started) in original Minigo repo to set up the environment.

Note that current implementation only works on Python3. We are working on Python2 compatibility.

## Training Model
One iteration of reinforcement learning consists of the following steps:
 - Bootstrap: initializes a random model
 - Selfplay: plays games with the latest model, producing data used for training
 - Gather: groups games played with the same model into larger files of tfexamples.
 - Train: trains a new model with the selfplay results from the most recent N
   generations.

 Run `minigo.py`.
 ```
 python3 minigo.py
 ```

## Validating Model
 Run `minigo.py` with `--holdout` argument
 ```
 python3 minigo.py --holdout=1
 ```
 The `--holdout` argument is to generate holdout dataset for model validation

## Evaluating MiniGo Models
 Run `minigo.py` with `--eval` argument
 ```
 python3 minigo.py --eval=1
 ```
 The `--eval` argument is to invoke the evaluation between the latest model and the current best model.

## Testing Pipeline
As the whole RL pipeline may takes hours to train even for a 9x9 board size, we provide a dummy model with a `--debug` mode for testing purpose.

 Run `minigo.py` with `--debug` argument
 ```
 python3 minigo.py --debug=1
 ```
 The `--debug` argument is for testing purpose with a dummy model.

Validation and evaluation can also be tested with the dummy model by combing their corresponding arguments with `--debug`.
To test validation, run the following commands:
 ```
 python3 minigo.py --debug=1 --validation=1
 ```
To test evaluation, run the following commands:
 ```
 python3 minigo.py --debug=1 --evaluation=1
 ```
To test both validation and evaluation, run the following commands:
 ```
 python3 minigo.py --debug=1 --validation=1 --eval=1
 ```

## MCTS and Go features (TODO)
Code clean up on MCTS and Go features.
