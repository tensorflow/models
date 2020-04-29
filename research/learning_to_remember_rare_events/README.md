![No Maintenance Intended](https://img.shields.io/badge/No%20Maintenance%20Intended-%E2%9C%95-red.svg)
![TensorFlow Requirement: 1.x](https://img.shields.io/badge/TensorFlow%20Requirement-1.x-brightgreen)
![TensorFlow 2 Not Supported](https://img.shields.io/badge/TensorFlow%202%20Not%20Supported-%E2%9C%95-red.svg)

---

Code for the Memory Module as described
in "Learning to Remember Rare Events" by
Lukasz Kaiser, Ofir Nachum, Aurko Roy, and Samy Bengio
published as a conference paper at ICLR 2017.

Requirements:
* TensorFlow (see tensorflow.org for how to install)
* Some basic command-line utilities (git, unzip).

Description:

The general memory module is located in memory.py.
Some code is provided to see the memory module in
action on the standard Omniglot dataset.
Download and setup the dataset using data_utils.py
and then run the training script train.py
(see example commands below).

Note that the structure and parameters of the model
are optimized for the data preparation as provided.

Quick Start:

First download and set-up Omniglot data by running

```
python data_utils.py
```

Then run the training script:

```
python train.py --memory_size=8192 \
  --batch_size=16 --validation_length=50 \
  --episode_width=5 --episode_length=30
```

The first validation batch may look like this (although it is noisy):
```
0-shot: 0.040, 1-shot: 0.404, 2-shot: 0.516, 3-shot: 0.604,
  4-shot: 0.656, 5-shot: 0.684
```
At step 500 you may see something like this:
```
0-shot: 0.036, 1-shot: 0.836, 2-shot: 0.900, 3-shot: 0.940,
  4-shot: 0.944, 5-shot: 0.916
```
At step 4000 you may see something like this:
```
0-shot: 0.044, 1-shot: 0.960, 2-shot: 1.000, 3-shot: 0.988,
  4-shot: 0.972, 5-shot: 0.992
```

Maintained by Ofir Nachum (ofirnachum) and
Lukasz Kaiser (lukaszkaiser).
