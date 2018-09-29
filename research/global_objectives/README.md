# Global Objectives
The Global Objectives library provides TensorFlow loss functions that optimize
directly for a variety of objectives including AUC, recall at precision, and
more. The global objectives losses can be used as drop-in replacements for
TensorFlow's standard multilabel loss functions:
`tf.nn.sigmoid_cross_entropy_with_logits` and `tf.losses.sigmoid_cross_entropy`.

Many machine learning classification models are optimized for classification
accuracy, when the real objective the user cares about is different and can be
precision at a fixed recall, precision-recall AUC, ROC AUC or similar metrics.
These are referred to as "global objectives" because they depend on how the
model classifies the dataset as a whole and do not decouple across data points
as accuracy does.

Because these objectives are combinatorial, discontinuous, and essentially
intractable to optimize directly, the functions in this library approximate
their corresponding objectives. This approximation approach follows the same
pattern as optimizing for accuracy, where a surrogate objective such as
cross-entropy or the hinge loss is used as an upper bound on the error rate.

## Getting Started
For a full example of how to use the loss functions in practice, see
loss_layers_example.py.

Briefly, global objective losses can be used to replace
`tf.nn.sigmoid_cross_entropy_with_logits` by providing the relevant
additional arguments. For example,

``` python
tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
```

could be replaced with

``` python
global_objectives.recall_at_precision_loss(
    labels=labels,
    logits=logits,
    target_precision=0.95)[0]
```

Just as minimizing the cross-entropy loss will maximize accuracy, the loss
functions in loss_layers.py were written so that minimizing the loss will
maximize the corresponding objective.

The global objective losses have two return values -- the loss tensor and
additional quantities for debugging and customization -- which is why the first
value is used above. For more information, see
[Visualization & Debugging](#visualization-debugging).

## Binary Label Format
Binary classification problems can be represented as a multi-class problem with
two classes, or as a multi-label problem with one label. (Recall that multiclass
problems have mutually exclusive classes, e.g. 'cat xor dog', and multilabel
have classes which are not mutually exclusive, e.g. an image can contain a cat,
a dog, both, or neither.) The softmax loss
(`tf.nn.softmax_cross_entropy_with_logits`) is used for multi-class problems,
while the sigmoid loss (`tf.nn.sigmoid_cross_entropy_with_logits`) is used for
multi-label problems.

A multiclass label format for binary classification might represent positives
with the label [1, 0] and negatives with the label [0, 1], while the multilbel
format for the same problem would use [1] and [0], respectively.

All global objectives loss functions assume that the multilabel format is used.
Accordingly, if your current loss function is softmax, the labels will have to
be reformatted for the loss to work properly.

## Dual Variables
Global objectives losses (except for `roc_auc_loss`) use internal variables
called dual variables or Lagrange multipliers to enforce the desired constraint
(e.g. if optimzing for recall at precision, the constraint is on precision).

These dual variables are created and initialized internally by the loss
functions, and are updated during training by the same optimizer used for the
model's other variables. To initialize the dual variables to a particular value,
use the `lambdas_initializer` argument. The dual variables can be found under
the key `lambdas` in the `other_outputs` dictionary returned by the losses.

## Loss Function Arguments
The following arguments are common to all loss functions in the library, and are
either required or very important.

* `labels`: Corresponds directly to the `labels` argument of
  `tf.nn.sigmoid_cross_entropy_with_logits`.
* `logits`: Corresponds directly to the `logits` argument of
  `tf.nn.sigmoid_cross_entropy_with_logits`.
* `dual_rate_factor`: A floating point value which controls the step size for
  the Lagrange multipliers. Setting this value less than 1.0 will cause the
  constraint to be enforced more gradually and will result in more stable
  training.

In addition, the objectives with a single constraint (e.g.
`recall_at_precision_loss`) have an argument (e.g. `target_precision`) used to
specify the value of the constraint. The optional `precision_range` argument to
`precision_recall_auc_loss` is used to specify the range of precision values
over which to optimize the AUC, and defaults to the interval [0, 1].

Optional arguments:

* `weights`: A tensor which acts as coefficients for the loss. If a weight of x
  is provided for a datapoint and that datapoint is a true (false) positive
  (negative), it will be counted as x true (false) positives (negatives).
  Defaults to 1.0.
* `label_priors`: A tensor specifying the fraction of positive datapoints for
  each label. If not provided, it will be computed inside the loss function.
* `surrogate_type`: Either 'xent' or 'hinge', specifying which upper bound
      should be used for indicator functions.
* `lambdas_initializer`: An initializer for the dual variables (Lagrange
  multipliers). See also the Dual Variables section.
* `num_anchors` (precision_recall_auc_loss only): The number of grid points used
  when approximating the AUC as a Riemann sum.

## Hyperparameters
While the functional form of the global objectives losses allow them to be
easily substituted in place of `sigmoid_cross_entropy_with_logits`, model
hyperparameters such as learning rate, weight decay, etc. may need to be
fine-tuned to the new loss. Fortunately, the amount of hyperparameter re-tuning
is usually minor.

The most important hyperparameters to modify are the learning rate and
dual_rate_factor (see the section on Loss Function Arguments, above).

## Visualization & Debugging
The global objectives losses return two values. The first is a tensor
representing the numerical value of the loss, which can be passed to an
optimizer. The second is a dictionary of tensors created by the loss function
which are not necessary for optimization but useful in debugging. These vary
depending on the loss function, but usually include `lambdas` (the Lagrange
multipliers) as well as the lower bound on true positives and upper bound on
false positives.

When visualizing the loss during training, note that the global objectives
losses differ from standard losses in some important ways:

* The global losses may be negative. This is because the value returned by the
  loss includes terms involving the Lagrange multipliers, which may be negative.
* The global losses may not decrease over the course of training. To enforce the
  constraints in the objective, the loss changes over time and may increase.

## More Info
For more details, see the [Global Objectives paper](https://arxiv.org/abs/1608.04802).

## Maintainers

* Mariano Schain
* Elad Eban
* [Alan Mackey](https://github.com/mackeya-google)
