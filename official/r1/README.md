# Legacy Models Collection

The R1 folder contains legacy model implmentation and models that will not
update to TensorFlow 2.x. They do not have solid performance tracking.

## Legacy model implmentation

Transformer and MNIST implementation uses pure TF 1.x TF-Estimator.
Users should follow the corresponding TF 2.x implmentation inside the
official model garden.

## Models that will not update to TensorFlow 2.x

*   [boosted_trees](boosted_trees): A Gradient Boosted Trees model to
    classify higgs boson process from HIGGS Data Set.
*   [wide_deep](wide_deep): A model that combines a wide model and deep
    network to classify census income data.
