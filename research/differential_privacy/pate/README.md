Implementation of an RDP privacy accountant and smooth sensitivity analysis for
the PATE framework. The underlying theory and supporting experiments appear in
"Scalable Private Learning with PATE" by Nicolas Papernot, Shuang Song, Ilya
Mironov, Ananth Raghunathan, Kunal Talwar, Ulfar Erlingsson (ICLR 2018,
https://arxiv.org/abs/1802.08908).

## Overview

The PATE ('Private Aggregation of Teacher Ensembles') framework was introduced 
by Papernot et al. in "Semi-supervised Knowledge Transfer for Deep Learning from
Private Training Data" (ICLR 2017, https://arxiv.org/abs/1610.05755). The 
framework enables model-agnostic training that provably provides [differential
privacy](https://en.wikipedia.org/wiki/Differential_privacy) of the training 
dataset. 

The framework consists of _teachers_, the _student_ model, and the _aggregator_. The 
teachers are models trained on disjoint subsets of the training datasets. The student
model has access to an insensitive (e.g., public) unlabelled dataset, which is labelled by 
interacting with the ensemble of teachers via the _aggregator_. The aggregator tallies 
outputs of the teacher models, and either forwards a (noisy) aggregate to the student, or
refuses to answer.

Differential privacy is enforced by the aggregator. The privacy guarantees can be _data-independent_,
which means that they are solely the function of the aggregator's parameters. Alternatively, privacy 
analysis can be _data-dependent_, which allows for finer reasoning where, under certain conditions on
the input distribution, the final privacy guarantees can be improved relative to the data-independent
analysis. Data-dependent privacy guarantees may, by themselves, be a function of sensitive data and 
therefore publishing these guarantees requires its own sanitization procedure. In our case 
sanitization of data-dependent privacy guarantees proceeds via _smooth sensitivity_ analysis.

The common machinery used for all privacy analyses in this repository is the 
R&eacute;nyi differential privacy, or RDP (see https://arxiv.org/abs/1702.07476). 

This repository contains implementations of privacy accountants and smooth 
sensitivity analysis for several data-independent and data-dependent mechanism that together
comprise the PATE framework.


### Requirements

* Python, version &ge; 2.7
* absl (see [here](https://github.com/abseil/abseil-py), or just type `pip install absl-py`)
* numpy
* scipy
* sympy (for smooth sensitivity analysis)
* unittest (for testing)


### Self-testing

To verify the installation run
```bash
$ python core_test.py
$ python smooth_sensitivity_test.py
```


## Files in this directory

*   core.py &mdash; RDP privacy accountant for several vote aggregators (GNMax,
    Threshold, Laplace).

*   smooth_sensitivity.py &mdash; Smooth sensitivity analysis for GNMax and
    Threshold mechanisms.

*   core_test.py and smooth_sensitivity_test.py &mdash; Unit tests for the
    files above.

## Contact information

You may direct your comments to mironov@google.com and PR to @ilyamironov.
