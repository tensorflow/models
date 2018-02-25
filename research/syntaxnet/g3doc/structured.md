# Structured training with TensorFlow

[TOC]

## Overview

The general goal is to train structured models, where there's a beam or a
trellis, normalized globally.

To begin, we implement the CRF objective (also termend "contrastive learning")
for parsing. Paraphrasing the notes of who/mjcollins, we have globally
normalized path probabilities

$$p_b(k) = \frac{\exp \sum_{j=1}^l A_b(j, K_b(j,k), D_b(j,k))}{\sum_{k=1}^B \exp
\sum_{j=1}^l A_b(j, K_b(j,k), D_b(j,k)) } ,$$

where \\(A_b(j, k, m)\\) is the activation for step \\(j\\), beam \\(b\\), slot
\\(k\\), and output class \\(m\\). We also have \\(D_b(j,k)\\), the decision at
step \\(j\\) in the final hypothesis \\(k\\), and \\(K_b(j,k)\\), the position
at step \\(j\\) of final hypothesis \\(k\\).

The final log-likelihood is then:

$$\log p_b(g_b) = \sum_{j=1}^l A_b(j, K_b(j,g_b), D_b(j,g_b)) - \log \sum_{k=1}^B \exp
\sum_{j=1}^l A_b(j, K_b(j,k), D_b(j,k)) ,$$

where \\(g_b\\) is the position of the gold hypothesis in the final beam.

With early updates, the 'final' beam is the one where the gold would have fallen
off the beam. With this option, using a large beam and setting the maximum path
length to one results in effectively doing the usual non-structured
training. This is a useful control.

## Strategy in TensorFlow

The beam reader Ops take care of producing \\(K_b\\), \\(D_b\\) and \\(g_b\\),
while the TensorFlow graph takes care of constructing \\(A_b\\).

The only fixed dimension of the tensor-like object \\(A_b\\) is its final
dimension, which is always the number of classes. Everything else, including the
size of the beam and the number of steps, can vary. And due to early updates
even the effective number of batches can shrink as the gold falls off.

Therefore \\(A\\) is stored as matrix, with a fixed number of columns but a
variable number of rows. The sparse storage of \\(A\\), named `concat_scores` in
the diagram below and in the code, relies of keeping track of the (variable)
number of beams and beam slots.

Due to early updates we do not know ahead of time the number of steps that need
to be taken before all the gold falls off. Therefore we use TensorFlow `Cond`
operators to terminate parsing when all the gold paths have fallen off.

## Diagrams

![Sparse scores matrix](https://docs.google.com/drawings/d/1ursvLgMU3QBzw2DVMk5_UWXp5-LAT04Oc454Wgvkk1g/export/png "Sparse scores matrix")

## Provided Ops

*  BeamParseReader opens the corpus for reading and allocates the main
   BeamState object that will be used by the other Ops. It outputs a handle to
   the BeamState and the feature vectors associated with the first token. Only
   one BeamParseReader needs to be created per decoding network.

*  BeamParser takes the scores output by the network and combines it with
   the BeamState to generate an updated beam and a fresh set of feature
   vectors. The BeamState object is mutated in the process.  Multiple
   BeamParsers are chained together to construct a fully unrolled
   network. It is important that consecutive BeamParsers have the correct
   dependencies to ensure that only one BeamParser is evaluated at any time.

*  BeamParserOutput processes the BeamState to generate the paths over which
   the cost will be computed. Multiple BeamParserOutputs can be
   instantiated, although they all access the same underlying BeamState, so in
   practice only one is needed.

*  BeamEvalOutput processes the BeamState to compute eval metrics for the
   best path in each beam. The eval metrics is returned as a vector of length
   two, containing [#tokens, #correct tokens].
