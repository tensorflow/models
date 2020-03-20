# Legacy Models Collection

The R1 folder contains legacy model implmentation and models that will not
update to TensorFlow 2.x. They do not have solid performance tracking.

**Note: models will be removed from the master branch by 2020/06.**

After removal, you can still access to these legacy models in the previous
released tags, e.g. [v2.1.0](https://github.com/tensorflow/models/releases/tag/v2.1.0).


## Legacy model implmentation

Transformer and MNIST implementation uses pure TF 1.x TF-Estimator.
Users should follow the corresponding TF 2.x implmentation inside the
official model garden.

## Models that will not update to TensorFlow 2.x

*   [boosted_trees](boosted_trees): A Gradient Boosted Trees model to
    classify higgs boson process from HIGGS Data Set.
*   [wide_deep](wide_deep): A model that combines a wide model and deep
    network to classify census income data.
