<font size=4><b>Deep Learning with Differential Privacy</b></font>

Open Sourced By: Xin Pan (xpan@google.com, github: panyx0718)


###Introduction for dp_sgd/README.md

Machine learning techniques based on neural networks are achieving remarkable 
results in a wide variety of domains. Often, the training of models requires 
large, representative datasets, which may be crowdsourced and contain sensitive 
information. The models should not expose private information in these datasets. 
Addressing this goal, we develop new algorithmic techniques for learning and a 
refined analysis of privacy costs within the framework of differential privacy. 
Our implementation and experiments demonstrate that we can train deep neural 
networks with non-convex objectives, under a modest privacy budget, and at a 
manageable cost in software complexity, training efficiency, and model quality.

paper: https://arxiv.org/abs/1607.00133


###Introduction for multiple_teachers/README.md

This repository contains code to create a setup for learning privacy-preserving 
student models by transferring knowledge from an ensemble of teachers trained 
on disjoint subsets of the data for which privacy guarantees are to be provided.

Knowledge acquired by teachers is transferred to the student in a differentially
private manner by noisily aggregating the teacher decisions before feeding them
to the student during training.

paper: https://arxiv.org/abs/1610.05755
