![No Maintenance Intended](https://img.shields.io/badge/No%20Maintenance%20Intended-%E2%9C%95-red.svg)
![TensorFlow Requirement: 1.x](https://img.shields.io/badge/TensorFlow%20Requirement-1.x-brightgreen)
![TensorFlow 2 Not Supported](https://img.shields.io/badge/TensorFlow%202%20Not%20Supported-%E2%9C%95-red.svg)

# Stochastic Ensemble Value Expansion

*A hybrid model-based/model-free reinforcement learning algorithm for sample-efficient continuous control.*

This is the code repository accompanying the paper Sample-Efficient Reinforcement Learning with
Stochastic Ensemble Value Expansion, by Buckman et al. (2018).

#### Abstract:
Merging model-free and model-based approaches in reinforcement learning has the potential to achieve
the high performance of model-free algorithms with low sample complexity. This is difficult because
an imperfect dynamics model can degrade the performance of the learning algorithm, and in sufficiently
complex environments, the dynamics model will always be imperfect. As a result, a key challenge is to
combine model-based approaches with model-free learning in such a way that errors in the model do not
degrade performance. We propose *stochastic ensemble value expansion* (STEVE), a novel model-based
technique that addresses this issue. By dynamically interpolating between model rollouts of various horizon
lengths for each individual example, STEVE ensures that the model is only utilized when doing so does not
introduce significant errors. Our approach outperforms model-free baselines on challenging continuous
control benchmarks with an order-of-magnitude increase in sample efficiency, and in contrast to previous
model-based approaches, performance does not degrade as the environment gets more complex.

## Installation
This code is compatible with Ubuntu 16.04 and Python 2.7. There are several prerequisites:
*    Numpy, Scipy, and Portalocker: `pip install numpy scipy portalocker`
*    TensorFlow 1.6 or above. Instructions can be found on the official TensorFlow page:
     [https://www.tensorflow.org/install/install_linux](https://www.tensorflow.org/install/install_linux).
     We suggest installing the GPU version of TensorFlow to speed up training.
*    OpenAI Gym version 0.9.4. Instructions can be found in the OpenAI Gym repository:
     [https://github.com/openai/gym#installation](https://github.com/openai/gym#installation).
     Note that you need to replace "pip install gym[all]" with "pip install gym[all]==0.9.4", which
     will ensure that you get the correct version of Gym. (The current version of Gym has deprecated
     the -v1 MuJoCo environments, which are the environments studied in this paper.)
*    MuJoCo version 1.31, which can be downloaded here: [https://www.roboti.us/download/mjpro131_linux.zip](https://www.roboti.us/download/mjpro131_linux.zip).
     Simply run: ```
     cd ~; mkdir -p .mujoco; cd .mujoco/; wget https://www.roboti.us/download/mjpro131_linux.zip; unzip mjpro131_linux.zip```
     You also need to get a license, and put the license key in ~/.mujoco/ as well.
*    Optionally, Roboschool version 1.1. This is needed only to replicate the Roboschool experiments.
     Instructions can be found in the OpenAI Roboschool repository:
     [https://github.com/openai/roboschool#installation](https://github.com/openai/roboschool#installation).
*    Optionally, MoviePy to render trained agents. Instructions on the MoviePy homepage:
     [https://zulko.github.io/moviepy/install.html](https://zulko.github.io/moviepy/install.html).

## Running Experiments
To run an experiment, run master.py and pass in a config file and GPU ID. For example: ```
python master.py config/experiments/speedruns/humanoid/speedy_steve0.json 0```
The `config/experiments/`
directory contains configuration files for all of the experiments run in the paper.

The GPU ID specifies the GPU that should be used to learn the policy. For model-based approaches, the
next GPU (i.e. GPU_ID+1) is used to learn the worldmodel in parallel.

To resume an experiment that was interrupted, use the same config file and pass the `--resume` flag: ```
python master.py config/experiments/speedruns/humanoid/speedy_steve0.json 0 --resume```

## Output
For each experiment, two folders are created in the output directory: `<ENVIRONMENT>/<EXPERIMENT>/log`
and `<ENVIRONMENT>/<EXPERIMENT>/checkpoints`. The log directory contains the following:

*  `hps.json` contains the accumulated hyperparameters of the config file used to generate these results
*  `valuerl.log` and `worldmodel.log` contain the log output of the learners. `worldmodel.log` will not
   exist if you are not learning a worldmodel.
*  `<EXPERIMENT>.greedy.csv` records all of the scores of our evaluators. The four columns contain time (hours),
   epochs, frames, and score.

The checkpoints directory contains the most recent versions of the policy and worldmodel, as well as checkpoints
of the policy, worldmodel, and their respective replay buffers at various points throughout training.

## Code Organization
`master.py` launches four types of processes: a ValueRlLearner to learn the policy, a WorldmodelLearner
to learn the dynamics model, several Interactors to gather data from the environment to train on, and
a few Evaluators to run the greedy policy in the environment and record the score.

`learner.py` contains a general framework for models which learn from a replay buffer. This is where
most of the code for the overall training loop is located. `valuerl_learner.py` and `worldmodel_learner.py`
contain a small amount of model-specific training loop code.

`valuerl.py` implements the core model for all value-function-based policy learning techniques studied
in the paper, including DDPG, MVE, STEVE, etc. Similarly, `worldmodel.py` contains the core model for
our dynamics model and reward function.

`replay.py` contains the code for the replay buffer. `nn.py`, `envwrap.py`, `config.py`, and `util.py`
each contain various helper functions.

`toy_demo.py` is a self-contained demo, written in numpy, that was used to generate the results for the
toy examples in the first segment of the paper.

`visualizer.py` is a utility script for loading trained policies and inspecting them. In addition to a
config file and a GPU, it takes the filename of the model to load as a mandatory third argument.

## Contact
Please contact GitHub user buckman-google (jacobbuckman@gmail.com) with any questions.
