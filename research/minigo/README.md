# MiniGo
This is a simplified implementation of MiniGo based on the code provided by the authors: [MiniGo](https://github.com/tensorflow/minigo).

MiniGo is a minimalist Go engine modeled after AlphaGo Zero, ["Mastering the Game of Go without Human
Knowledge"](https://www.nature.com/articles/nature24270). An useful one-diagram overview of Alphago Zero can be found in the [cheat sheet](https://medium.com/applied-data-science/alphago-zero-explained-in-one-diagram-365f5abf67e0).

The implementation of MiniGo consists of three main components: the DualNet model, the Monte Carlo Tree Search (MCTS), and Go domain knowledge. Currently, the **DualNet model** is our focus.


## DualNet Architecture
DualNet is the neural network used in MiniGo. It's based on residual blocks with two heads output. Following is a brief overview of the DualNet architecture.

### Input Features
The input to the neural network is a [board_size * board_size * 17] image stack
comprising 17 binary feature planes. 8 feature planes consist of binary values
indicating the presence of the current player's stones; A further 8 feature
planes represent the corresponding features for the opponent's stones; The final
feature plane represents the color to play, and has a constant value of either 1
if black is to play or 0 if white to play. Check [features.py](features.py) for more details.

### Neural Network Structure
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

Note: num_filter is 128 for 19 x 19 board size, and 32 for 9 x 9 board size in MiniGo implementation.

### Dual Heads Output
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

In MiniGo, the overall network depth, in the 10 or 20 block network, is 19 or 39
parameterized layers respectively for the residual tower, plus an additional 2
layers for the policy head and 3 layers for the value head.

## Getting Started
This project assumes you have virtualenv, TensorFlow (>= 1.5) and two other Go-related
packages pygtp(>=0.4) and sgf (==0.5).

## Training Model
One iteration of reinforcement learning (RL) consists of the following steps:
 - Bootstrap: initializes a random DualNet model. If the estimator directory has exist, the model is initialized with the last checkpoint.
 - Selfplay: plays games with the latest model or the best model so far identified by evaluation, producing data used for training
 - Gather: groups games played with the same model into larger files of tfexamples.
 - Train: trains a new model with the selfplay results from the most recent N generations.

To run the RL pipeline, issue the following command:
 ```
 python minigo.py --base_dir=$HOME/minigo/ --board_size=9 --batch_size=256
 ```
 Arguments:
   * `--base_dir`: Base directory for MiniGo data and models. If not specified, it's set as /tmp/minigo/ by default.
   * `--board_size`: Go board size. It can be either 9 or 19. By default, it's 9.
   * `--batch_size`: Batch size for model training. If not specified, it's calculated based on go board size.
 Use the `--help` or `-h` flag to get a full list of possible arguments. Besides all these arguments, other parameters about RL pipeline and DualNet model can be found and configured in [model_params.py](model_params.py).

Suppose the base directory argument `base_dir` is `$HOME/minigo/` and we use 9 as the `board_size`. After model training, the following directories are created to store models and game data:

    $HOME/minigo                  # base directory
    │
    ├── 9_size                    # directory for 9x9 board size
    │   │
    │   ├── data
    │   │   ├── holdout           # holdout data for model validation
    │   │   ├── selfplay          # data generated by selfplay of each model
    │   │   └── training_chunks   # gatherd tf_examples for model training
    │   │
    │   ├── estimator_model_dir   # estimator working directory
    │   │
    │   ├── trained_models        # all the trained models
    │   │
    │   └── sgf                   # sgf (smart go files) folder
    │       ├── 000000-bootstrap  # model name
    │       │      ├── clean      # clean sgf files of model selfplay
    │       │      └── full       # full sgf files of model selfplay
    │       ├── ...
    │       └── evaluate          # clean sgf files of model evaluation
    │
    └── ...

## Validating Model
To validate the trained model, issue the following command with `--validation` argument:
 ```
 python minigo.py --base_dir=$HOME/minigo/ --board_size=9 --batch_size=256 --validation
 ```

## Evaluating Models
The performance of two models are compared with evaluation step. Given two models, one plays black and the other plays white. They play several games (# of games can be configured by parameter `eval_games` in [model_params.py](model_params.py)), and the one wins by a margin of 55% will be the winner.

To include the evaluation step in the RL pipeline, `--evaluation` argument can be specified to compare the performance of the `current_trained_model` and the `best_model_so_far`. The winner is used to update `best_model_so_far`. Run the following command to include evaluation step in the pipeline:
 ```
 python minigo.py --base_dir=$HOME/minigo/ --board_size=9 --batch_size=256 --evaluation
 ```

## Testing Pipeline
As the whole RL pipeline may take hours to train even for a 9x9 board size, a `--test` argument is provided to test the pipeline quickly with a dummy neural network model.

To test the RL pipeline with a dummy model, issue the following command:
```
 python minigo.py --base_dir=$HOME/minigo/ --board_size=9 --batch_size=256 --test
```

## Running Self-play Only
Self-play only option is provided to run selfplay step individually to generate training data in parallel. Issue the following command to run selfplay only with the latest trained model:
```
 python minigo.py --selfplay
```
Other optional arguments:
   * `--selfplay_model_name`: The name of the model used for selfplay only. If not specified, the latest trained model will be used for selfplay.
   * `--selfplay_max_games`: The maximum number of games selfplay is required to generate. If not specified, the default parameter `max_games_per_generation` is used.
