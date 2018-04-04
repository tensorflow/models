# Learning Unsupervised Learning Rules
This repository contains code and weights for the learned update rule
presented in "Learning Unsupervised Learning Rules." At this time, this
code can not meta-train the update rule.


### Structure
`run_eval.py` contains the main training loop. This constructs an op
that runs one iteration of the learned update rule and assigns the
results to variables. Additionally, it loads the weights from our
pre-trained model.

The base model and the update rule architecture definition can be found in
`architectures/more_local_weight_update.py`. For a complete description
of the model, see our [paper](https://arxiv.org/abs/1804.00222).

### Dependencies
[absl]([https://github.com/abseil/abseil-py), [tensorflow](https://tensorflow.org), [sonnet](https://github.com/deepmind/sonnet)

### Usage

First, download the [pre-trained optimizer model weights](https://storage.googleapis.com/learning_unsupervised_learning/200_tf_graph.zip) and extract it.

```bash
# move to the folder above this folder
cd path_to/research/learning_unsupervised_learning/../

# launch the eval script
python -m learning_unsupervised_learning.run_eval \
--train_log_dir="/tmp/learning_unsupervised_learning" \
--checkpoint_dir="/path/to/downloaded/model/tf_graph_data.ckpt"
```

### Contact
Luke Metz, Niru Maheswaranathan, Github: @lukemetz, @nirum. Email: {lmetz, nirum}@google.com


