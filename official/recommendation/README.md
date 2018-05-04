# Recommendation Model
This is an implementation of the Neural Collaborative Filtering (NCF) framework with Neural Matrix Factorization (NeuMF) model as described in the [Neural Collaborative Filtering](https://arxiv.org/abs/1708.05031) paper. Current implementation is based on the code from the authors' [NCF code](https://github.com/hexiangnan/neural_collaborative_filtering) and the Standford pytorch implementation at [mlperf repo](https://github.com/mlperf/reference/tree/master/recommendation/pytorch).

NCF is a general framework under which a neural network architecture is proposed to model latent features of users and items in collaborative filtering of recommendation. Unlike traditional models, NCF does not resort to Matrix Factorization (MF) with an inner product on latent features of users and items. It replaces the inner product with a multi-layer perceptron that can learn an arbitrary function from data.

Two instantiations of NCF are Generalized Matrix Factorization (GMF) and Multi-Layer Perceptron (MLP). GMF applies a linear kernel to model the latent feature interactions, and and MLP uses a nonlinear kernel to learn the interaction function from data. NeuMF is a fused model of GMF and MLP to better model the complex user-item interactions, and unifies the strengths of linearity of MF and non-linearity of MLP for modeling the user-item latent structures. NeuMF allows GMF and MLP to learn separate embeddings, and combines the two models by concatenating their last hidden layer.

## Overview (TODO)

### NeuMF Model Architecture





### Dataset





## Running Code (TODO)

### Download and Preprocess Dataset
   Command to run:
   ```
   python data_download.py
   ```


### Train and Evaluate Model
   Command to run:
   ```
   python ncf.py
   ```
