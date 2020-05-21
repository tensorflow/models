![No Maintenance Intended](https://img.shields.io/badge/No%20Maintenance%20Intended-%E2%9C%95-red.svg)
![TensorFlow Requirement: 1.x](https://img.shields.io/badge/TensorFlow%20Requirement-1.x-brightgreen)
![TensorFlow 2 Not Supported](https://img.shields.io/badge/TensorFlow%202%20Not%20Supported-%E2%9C%95-red.svg)

# Deep Bayesian Bandits Library

This library corresponds to the *[Deep Bayesian Bandits Showdown: An Empirical
Comparison of Bayesian Deep Networks for Thompson
Sampling](https://arxiv.org/abs/1802.09127)* paper, published in
[ICLR](https://iclr.cc/) 2018. We provide a benchmark to test decision-making
algorithms for contextual-bandits. In particular, the current library implements
a variety of algorithms (many of them based on approximate Bayesian Neural
Networks and Thompson sampling), and a number of real and syntethic data
problems exhibiting a diverse set of properties.

It is a Python library that uses [TensorFlow](https://www.tensorflow.org/).

We encourage contributors to add new approximate Bayesian Neural Networks or,
more generally, contextual bandits algorithms to the library. Also, we would
like to extend the data sources over time, so we warmly encourage contributions
in this front too!

Please, use the following when citing the code or the paper:

```
@article{riquelme2018deep, title={Deep Bayesian Bandits Showdown: An Empirical
Comparison of Bayesian Deep Networks for Thompson Sampling},
author={Riquelme, Carlos and Tucker, George and Snoek, Jasper},
journal={International Conference on Learning Representations, ICLR.}, year={2018}}
```

**Contact**. This repository is maintained by [Carlos Riquelme](http://rikel.me) ([rikel](https://github.com/rikel)). Feel free to reach out directly at [rikel@google.com](mailto:rikel@google.com) with any questions or comments.


We first briefly introduce contextual bandits, Thompson sampling, enumerate the
implemented algorithms, and the available data sources. Then, we provide a
simple complete example illustrating how to use the library.

## Contextual Bandits

Contextual bandits are a rich decision-making framework where an algorithm has
to choose among a set of *k* actions at every time step *t*, after observing
a context (or side-information) denoted by *X<sub>t</sub>*. The general pseudocode for
the process if we use algorithm **A** is as follows:

```
At time t = 1, ..., T:
  1. Observe new context: X_t
  2. Choose action: a_t = A.action(X_t)
  3. Observe reward: r_t
  4. Update internal state of the algorithm: A.update((X_t, a_t, r_t))
```

The goal is to maximize the total sum of rewards: &sum;<sub>t</sub> r<sub>t</sub>

For example, each *X<sub>t</sub>* could encode the properties of a specific user (and
the time or day), and we may have to choose an ad, discount coupon, treatment,
hyper-parameters, or version of a website to show or provide to the user.
Hopefully, over time, we will learn how to match each type of user to the most
beneficial personalized action under some metric (the reward).

## Thompson Sampling

Thompson Sampling is a meta-algorithm that chooses an action for the contextual
bandit in a statistically efficient manner, simultaneously finding the best arm
while attempting to incur low cost. Informally speaking, we assume the expected
reward is given by some function
**E**[r<sub>t</sub> | X<sub>t</sub>, a<sub>t</sub>] = f(X<sub>t</sub>, a<sub>t</sub>).
Unfortunately, function **f** is unknown, as otherwise we could just choose the
action with highest expected value:
a<sub>t</sub><sup>*</sup> = arg max<sub>i</sub> f(X<sub>t</sub>, a<sub>t</sub>).

The idea behind Thompson Sampling is based on keeping a posterior distribution
&pi;<sub>t</sub> over functions in some family f &isin; F after observing the first
*t-1* datapoints. Then, at time *t*, we sample one potential explanation of
the underlying process: f<sub>t</sub> &sim; &pi;<sub>t</sub>, and act optimally (i.e., greedily)
*according to f<sub>t</sub>*. In other words, we choose
a<sub>t</sub> = arg max<sub>i</sub> f<sub>t</sub>(X<sub>t</sub>, a<sub>i</sub>).
Finally, we update our posterior distribution with the new collected
datapoint (X<sub>t</sub>, a<sub>t</sub>, r<sub>t</sub>).

The main issue is that keeping an updated posterior &pi;<sub>t</sub> (or, even,
sampling from it) is often intractable for highly parameterized models like deep
neural networks. The algorithms we list in the next section provide tractable
*approximations* that can be used in combination with Thompson Sampling to solve
the contextual bandit problem.

## Algorithms

The Deep Bayesian Bandits library includes the following algorithms (see the
[paper](https://arxiv.org/abs/1802.09127) for further details):

1.  **Linear Algorithms**. As a powerful baseline, we provide linear algorithms.
    In particular, we focus on the exact Bayesian linear regression
    implementation, while it is easy to derive the greedy OLS version (possibly,
    with epsilon-greedy exploration). The algorithm is implemented in
    *linear_full_posterior_sampling.py*, and it is instantiated as follows:

    ```
        linear_full = LinearFullPosteriorSampling('MyLinearTS', my_hparams)
    ```

2.  **Neural Linear**. We introduce an algorithm we call Neural Linear, which
    operates by learning a neural network to map contexts to rewards for each
    action, and ---simultaneously--- it updates a Bayesian linear regression in
    the last layer (i.e., the one that maps the final representation **z** to
    the rewards **r**). Thompson Sampling samples the linear parameters
    &beta;<sub>i</sub> for each action *i*, but keeps the network that computes the
    representation. Then, both parts (network and Bayesian linear regression)
    are updated, possibly at different frequencies. The algorithm is implemented
    in *neural_linear_sampling.py*, and we create an algorithm instance like
    this:

    ```
        neural_linear = NeuralLinearPosteriorSampling('MyNLinear', my_hparams)
    ```

3.  **Neural Greedy**. Another standard benchmark is to train a neural network
    that maps contexts to rewards, and at each time *t* just acts greedily
    according to the current model. In particular, this approach does *not*
    explicitly use Thompson Sampling. However, due to stochastic gradient
    descent, there is still some randomness in its output. It is
    straight-forward to add epsilon-greedy exploration to choose random
    actions with probability &epsilon; &isin; (0, 1). The algorithm is
    implemented in *neural_bandit_model.py*, and it is used together with
    *PosteriorBNNSampling* (defined in *posterior_bnn_sampling.py*) by calling:

    ```
      neural_greedy = PosteriorBNNSampling('MyNGreedy', my_hparams, 'RMSProp')
    ```

4.  **Stochastic Variational Inference**, Bayes by Backpropagation. We implement
    a Bayesian neural network by modeling each individual weight posterior as a
    univariate Gaussian distribution: w<sub>ij</sub> &sim; N(&mu;<sub>ij</sub>, &sigma;<sub>ij</sub><sup>2</sup>).
    Thompson sampling then samples a network at each time step
    by sampling each weight independently. The variational approach consists in
    maximizing a proxy for maximum likelihood of the observed data, the ELBO or
    variational lower bound, to fit the values of &mu;<sub>ij</sub>, &sigma;<sub>ij</sub><sup>2</sup>
    for every *i, j*.

    See [Weight Uncertainty in Neural
    Networks](https://arxiv.org/abs/1505.05424).

    The BNN algorithm is implemented in *variational_neural_bandit_model.py*,
    and it is used together with *PosteriorBNNSampling* (defined in
    *posterior_bnn_sampling.py*) by calling:

    ```
        bbb = PosteriorBNNSampling('myBBB', my_hparams, 'Variational')
    ```

5.  **Expectation-Propagation**, Black-box alpha-divergence minimization.
    The family of expectation-propagation algorithms is based on the message
    passing framework . They iteratively approximate the posterior by updating a
    single approximation factor (or site) at a time, which usually corresponds
    to the likelihood of one data point. We focus on methods that directly
    optimize the global EP objective via stochastic gradient descent, as, for
    instance, Power EP. For further details see original paper below.

    See [Black-box alpha-divergence
    Minimization](https://arxiv.org/abs/1511.03243).

    We create an instance of the algorithm like this:

    ```
        bb_adiv = PosteriorBNNSampling('MyEP', my_hparams, 'AlphaDiv')
    ```

6.  **Dropout**. Dropout is a training technique where the output of each neuron
    is independently zeroed out with probability *p* at each forward pass.
    Once the network has been trained, dropout can still be used to obtain a
    distribution of predictions for a specific input. Following the best action
    with respect to the random dropout prediction can be interpreted as an
    implicit form of Thompson sampling. The code for dropout is the same as for
    Neural Greedy (see above), but we need to set two hyper-parameters:
    *use_dropout=True* and *keep_prob=p* where *p* takes the desired value in
    (0, 1). Then:

    ```
        dropout = PosteriorBNNSampling('MyDropout', my_hparams, 'RMSProp')
    ```

7.  **Monte Carlo Methods**. To be added soon.

8.  **Bootstrapped Networks**. This algorithm trains simultaneously and in
    parallel **q** neural networks based on different datasets D<sub>1</sub>, ..., D<sub>q</sub>. The way those datasets are collected is by adding each new collected
    datapoint (X<sub>t</sub>, a<sub>t</sub>, r<sub>t</sub>) to each dataset *D<sub>i</sub>* independently and with
    probability p &isin; (0, 1]. Therefore, the main hyperparameters of the
    algorithm are **(q, p)**. In order to choose an action for a new context,
    one of the **q** networks is first selected with uniform probability (i.e.,
    *1/q*). Then, the best action according to the *selected* network is
    played.

    See [Deep Exploration via Bootstrapped
    DQN](https://arxiv.org/abs/1602.04621).

    The algorithm is implemented in *bootstrapped_bnn_sampling.py*, and we
    instantiate it as (where *my_hparams* contains both **q** and **p**):

    ```
        bootstrap = BootstrappedBNNSampling('MyBoot', my_hparams)
    ```

9.  **Parameter-Noise**. Another approach to approximate a distribution over
    neural networks (or more generally, models) that map contexts to rewards,
    consists in randomly perturbing a point estimate trained by Stochastic
    Gradient Descent on the data. The Parameter-Noise algorithm uses a heuristic
    to control the amount of noise &sigma;<sub>t</sub><sup>2</sup> it adds independently to the
    parameters representing a neural network: &theta;<sub>t</sub><sup>'</sup> = &theta;<sub>t</sub> + &epsilon; where
    &epsilon; &sim; N(0, &sigma;<sub>t</sub><sup>2</sup> Id).
    After using &theta;<sub>t</sub><sup>'</sup> for decision making, the following SGD
    training steps start again from &theta;<sub>t</sub>. The key hyperparameters to set
    are those controlling the noise heuristic.

    See [Parameter Space Noise for
    Exploration](https://arxiv.org/abs/1706.01905).

    The algorithm is implemented in *parameter_noise_sampling.py*, and we create
    an instance by calling:

    ```
        parameter_noise = ParameterNoiseSampling('MyParamNoise', my_hparams)
    ```

10. **Gaussian Processes**. Another standard benchmark are Gaussian Processes,
    see *Gaussian Processes for Machine Learning* by Rasmussen and Williams for
    an introduction. To model the expected reward of different actions, we fit a
    multitask GP.

    See [Multi-task Gaussian Process
    Prediction](http://papers.nips.cc/paper/3189-multi-task-gaussian-process-prediction.pdf).

    Our implementation is provided in *multitask_gp.py*, and it is instantiated
    as follows:

    ```
        gp = PosteriorBNNSampling('MyMultitaskGP', my_hparams, 'GP')
    ```

In the code snippet at the bottom, we show how to instantiate some of these
algorithms, and how to run the contextual bandit simulator, and display the
high-level results.

## Data

In the paper we use two types of contextual datasets: synthetic and based on
real-world data.

We provide functions that sample problems from those datasets. In the case of
real-world data, you first need to download the raw datasets, and pass the route
to the functions. Links for the datasets are provided below.

### Synthetic Datasets

Synthetic datasets are contained in the *synthetic_data_sampler.py* file. In
particular, it includes:

1.  **Linear data**. Provides a number of linear arms, and Gaussian contexts.

2.  **Sparse linear data**. Provides a number of sparse linear arms, and
    Gaussian contexts.

3.  **Wheel bandit data**. Provides sampled data from the wheel bandit data, see
    [Section 5.4](https://arxiv.org/abs/1802.09127) in the paper.

### Real-World Datasets

Real-world data generating functions are contained in the *data_sampler.py*
file.

In particular, it includes:

1.  **Mushroom data**. Each incoming context represents a different type of
    mushroom, and the actions are eat or no-eat. Eating an edible mushroom
    provides positive reward, while eating a poisonous one provides positive
    reward with probability *p*, and a large negative reward with probability
    *1-p*. All the rewards, and the value of *p* are customizable. The
    [dataset](https://archive.ics.uci.edu/ml/datasets/mushroom) is part of the
    UCI repository, and the bandit problem was proposed in Blundell et al.
    (2015). Data is available [here](https://storage.googleapis.com/bandits_datasets/mushroom.data)
    or alternatively [here](https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/),
    use the *agaricus-lepiota.data* file.

2.  **Stock data**. We created the Financial Dataset by pulling the stock prices
    of *d = 21* publicly traded companies in NYSE and Nasdaq, for the last 14
    years (*n = 3713*). For each day, the context was the price difference
    between the beginning and end of the session for each stock. We
    synthetically created the arms to be a linear combination of the contexts,
    representing *k = 8* different potential portfolios. Data is available
    [here](https://storage.googleapis.com/bandits_datasets/raw_stock_contexts).

3.  **Jester data**. We create a recommendation system bandit problem as
    follows. The Jester Dataset (Goldberg et al., 2001) provides continuous
    ratings in *[-10, 10]* for 100 jokes from a total of 73421 users. We find
    a *complete* subset of *n = 19181* users rating all 40 jokes. Following
    Riquelme et al. (2017), we take *d = 32* of the ratings as the context of
    the user, and *k = 8* as the arms. The agent recommends one joke, and
    obtains the reward corresponding to the rating of the user for the selected
    joke. Data is available [here](https://storage.googleapis.com/bandits_datasets/jester_data_40jokes_19181users.npy).

4.  **Statlog data**. The Shuttle Statlog Dataset (Asuncion & Newman, 2007)
    provides the value of *d = 9* indicators during a space shuttle flight,
    and the goal is to predict the state of the radiator subsystem of the
    shuttle. There are *k = 7* possible states, and if the agent selects the
    right state, then reward 1 is generated. Otherwise, the agent obtains no
    reward (*r = 0*). The most interesting aspect of the dataset is that one
    action is the optimal one in 80% of the cases, and some algorithms may
    commit to this action instead of further exploring. In this case, the number
    of contexts is *n = 43500*. Data is available [here](https://storage.googleapis.com/bandits_datasets/shuttle.trn) or alternatively
    [here](https://archive.ics.uci.edu/ml/datasets/Statlog+\(Shuttle\)), use
    *shuttle.trn* file.

5.  **Adult data**. The Adult Dataset (Kohavi, 1996; Asuncion & Newman, 2007)
    comprises personal information from the US Census Bureau database, and the
    standard prediction task is to determine if a person makes over 50K a year
    or not. However, we consider the *k = 14* different occupations as
    feasible actions, based on *d = 94* covariates (many of them binarized).
    As in previous datasets, the agent obtains a reward of 1 for making the
    right prediction, and 0 otherwise. The total number of contexts is *n =
    45222*. Data is available [here](https://storage.googleapis.com/bandits_datasets/adult.full) or alternatively
    [here](https://archive.ics.uci.edu/ml/datasets/adult), use *adult.data*
    file.

6.  **Census data**. The US Census (1990) Dataset (Asuncion & Newman, 2007)
    contains a number of personal features (age, native language, education...)
    which we summarize in *d = 389* covariates, including binary dummy
    variables for categorical features. Our goal again is to predict the
    occupation of the individual among *k = 9* classes. The agent obtains
    reward 1 for making the right prediction, and 0 otherwise. Data is available
    [here](https://storage.googleapis.com/bandits_datasets/USCensus1990.data.txt) or alternatively [here](https://archive.ics.uci.edu/ml/datasets/US+Census+Data+\(1990\)), use
    *USCensus1990.data.txt* file.

7.  **Covertype data**. The Covertype Dataset (Asuncion & Newman, 2007)
    classifies the cover type of northern Colorado forest areas in *k = 7*
    classes, based on *d = 54* features, including elevation, slope, aspect,
    and soil type. Again, the agent obtains reward 1 if the correct class is
    selected, and 0 otherwise. Data is available [here](https://storage.googleapis.com/bandits_datasets/covtype.data) or alternatively
    [here](https://archive.ics.uci.edu/ml/datasets/covertype), use
    *covtype.data* file.

In datasets 4-7, each feature of the dataset is normalized first.

## Usage: Basic Example

This library requires Tensorflow, Numpy, and Pandas.

The file *example_main.py* provides a complete example on how to use the
library. We run the code:

```
    python example_main.py
```

**Do not forget to** configure the routes to the data files at the top of *example_main.py*.

For example, we can run the Mushroom bandit for 2000 contexts on a few
algorithms as follows:

```
  # Problem parameters
  num_contexts = 2000

  # Choose data source among:
  # {linear, sparse_linear, mushroom, financial, jester,
  #  statlog, adult, covertype, census, wheel}
  data_type = 'mushroom'

  # Create dataset
  sampled_vals = sample_data(data_type, num_contexts)
  dataset, opt_rewards, opt_actions, num_actions, context_dim = sampled_vals

  # Define hyperparameters and algorithms
  hparams_linear = tf.contrib.training.HParams(num_actions=num_actions,
                                               context_dim=context_dim,
                                               a0=6,
                                               b0=6,
                                               lambda_prior=0.25,
                                               initial_pulls=2)

  hparams_dropout = tf.contrib.training.HParams(num_actions=num_actions,
                                                context_dim=context_dim,
                                                init_scale=0.3,
                                                activation=tf.nn.relu,
                                                layer_sizes=[50],
                                                batch_size=512,
                                                activate_decay=True,
                                                initial_lr=0.1,
                                                max_grad_norm=5.0,
                                                show_training=False,
                                                freq_summary=1000,
                                                buffer_s=-1,
                                                initial_pulls=2,
                                                optimizer='RMS',
                                                reset_lr=True,
                                                lr_decay_rate=0.5,
                                                training_freq=50,
                                                training_epochs=100,
                                                keep_prob=0.80,
                                                use_dropout=True)

  ### Create hyper-parameter configurations for other algorithms
    [...]

  algos = [
      UniformSampling('Uniform Sampling', hparams),
      PosteriorBNNSampling('Dropout', hparams_dropout, 'RMSProp'),
      PosteriorBNNSampling('BBB', hparams_bbb, 'Variational'),
      NeuralLinearPosteriorSampling('NeuralLinear', hparams_nlinear),
      LinearFullPosteriorSampling('LinFullPost', hparams_linear),
      BootstrappedBNNSampling('BootRMS', hparams_boot),
      ParameterNoiseSampling('ParamNoise', hparams_pnoise),
  ]

  # Run contextual bandit problem
  t_init = time.time()
  results = run_contextual_bandit(context_dim, num_actions, dataset, algos)
  _, h_rewards = results

  # Display results
  display_results(algos, opt_rewards, opt_actions, h_rewards, t_init, data_type)

```

The previous code leads to final results that look like:

```
---------------------------------------------------
---------------------------------------------------
mushroom bandit completed after 69.8401839733 seconds.
---------------------------------------------------
  0) LinFullPost         |               total reward =     4365.0.
  1) NeuralLinear        |               total reward =     4110.0.
  2) Dropout             |               total reward =     3430.0.
  3) ParamNoise          |               total reward =     3270.0.
  4) BootRMS             |               total reward =     3050.0.
  5) BBB                 |               total reward =     2505.0.
  6) Uniform Sampling    |               total reward =    -4930.0.
---------------------------------------------------
Optimal total reward = 5235.
Frequency of optimal actions (action, frequency):
[[0, 953], [1, 1047]]
---------------------------------------------------
---------------------------------------------------
```
