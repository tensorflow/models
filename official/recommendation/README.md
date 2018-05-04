# Recommendation Model
## Overview
This is an implementation of the Neural Collaborative Filtering (NCF) framework with Neural Matrix Factorization (NeuMF) model as described in the [Neural Collaborative Filtering](https://arxiv.org/abs/1708.05031) paper. Current implementation is based on the code from the authors' [NCF code](https://github.com/hexiangnan/neural_collaborative_filtering) and the Standford pytorch implementation at [mlperf repo](https://github.com/mlperf/reference/tree/master/recommendation/pytorch).

NCF is a general framework under which a neural network architecture is proposed to model latent features of users and items in collaborative filtering of recommendation. Unlike traditional models, NCF does not resort to Matrix Factorization (MF) with an inner product on latent features of users and items. It replaces the inner product with a multi-layer perceptron that can learn an arbitrary function from data.

Two instantiations of NCF are Generalized Matrix Factorization (GMF) and Multi-Layer Perceptron (MLP). GMF applies a linear kernel to model the latent feature interactions, and and MLP uses a nonlinear kernel to learn the interaction function from data. NeuMF is a fused model of GMF and MLP to better model the complex user-item interactions, and unifies the strengths of linearity of MF and non-linearity of MLP for modeling the user-item latent structures. NeuMF allows GMF and MLP to learn separate embeddings, and combines the two models by concatenating their last hidden layer. [neumf_model.py](neumf_model.py) defines the architecture details.


## Dataset
The [movielens datasets](http://files.grouplens.org/datasets/movielens/) are used for model training and evaluation. Specifically, we use two datasets: **ml-1m** and **ml-20m**. ml-1m contains 1,000,209 million anonymous ratings of approximately 3,706 movies
made by 6,040 users, and ml-20m contains 20,000,263 ratings of 26,744 movies by 138493 users. The ratings in the dataset are made on a 5-star scale (whole-star ratings only), and are transformed as implicit data where each entry is marked as 0 or 1 indicating whether the user has rated the movie.

## Running Code

### Download and Preprocess Dataset
To download the dataset and perform data preprocessing, issue the following command:
```
python data_download.py
```
Arguments:
  * `--data_dir`: Directory where to download and save the preprocessed data. By default, it is `/tmp/ml-data/`.
  * `--dataset`: The dataset name to be downloaded and preprocessed. By default, it is `ml-1m`.

Use the `--help` or `-h` flag to get a full list of possible arguments.

Note the ml-20m dataset is large (the rating file is ~500 MB), and it may take several minutes (<10 mins)for data preprocessing.

### Train and Evaluate Model
To train and evaluate the model, issue the following command:
```
python ncf.py
```
Arguments:
  * `--model_dir`: Directory to save model training checkpoints. By default, it is `/tmp/ncf/`.
  * `--data_dir`: This should be set to the same directory given to the `data_download`'s `data_dir` argument.
  * `--dataset`: The dataset name to be downloaded and preprocessed. By default, it is `ml-1m`.

There are other arguments about models and training process. Use the `--help` or `-h` flag to get a full list of possible arguments.
