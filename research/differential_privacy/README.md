# Deep Learning with Differential Privacy

Most of the content from this directory has moved to the [tensorflow/privacy](https://github.com/tensorflow/privacy) repository, which is dedicated to learning with (differential) privacy. The remaining code is related to the PATE papers from ICLR 2017 and 2018.

### Introduction for [multiple_teachers/README.md](multiple_teachers/README.md)

This repository contains code to create a setup for learning privacy-preserving 
student models by transferring knowledge from an ensemble of teachers trained 
on disjoint subsets of the data for which privacy guarantees are to be provided.

Knowledge acquired by teachers is transferred to the student in a differentially
private manner by noisily aggregating the teacher decisions before feeding them
to the student during training.

paper: https://arxiv.org/abs/1610.05755

### Introduction for [pate/README.md](pate/README.md)

Implementation of an RDP privacy accountant and smooth sensitivity analysis for the PATE framework. The underlying theory and supporting experiments appear in "Scalable Private Learning with PATE" by Nicolas Papernot, Shuang Song, Ilya Mironov, Ananth Raghunathan, Kunal Talwar, Ulfar Erlingsson (ICLR 2018) 

paper: https://arxiv.org/abs/1802.08908

