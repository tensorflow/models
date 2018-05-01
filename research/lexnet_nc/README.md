# LexNET for Noun Compound Relation Classification

This is a [Tensorflow](http://www.tensorflow.org/) implementation of the LexNET
algorithm for classifying relationships, specifically applied to classifying the
relationships that hold between noun compounds:

* *olive oil* is oil that is *made from* olives
* *cooking oil* which is oil that is *used for* cooking
* *motor oil* is oil that is *contained in* a motor

The model is a supervised classifier that predicts the relationship that holds
between the constituents of a two-word noun compound using:

1. A neural "paraphrase" of each syntactic dependency path that connects the
   constituents in a large corpus. For example, given a sentence like *This fine
   oil is made from first-press olives*, the dependency path is something like
   `oil <NSUBJPASS made PREP> from POBJ> olive`.
2. The distributional information provided by the individual words; i.e., the
   word embeddings of the two consituents.
3. The distributional signal provided by the compound itself; i.e., the
   embedding of the noun compound in context.

The model includes several variants: *path-based model* uses (1) alone, the
*distributional model* uses (2) alone, and the *integrated model* uses (1) and
(2).  The *distributional-nc model* and the *integrated-nc* model each add (3).

Training a model requires the following:

1. A collection of noun compounds that have been labeled using a *relation
   inventory*.  The inventory describes the specific relationships that you'd
   like the model to differentiate (e.g. *part of* versus *composed of* versus
   *purpose*), and generally may consist of tens of classes. 
   You can download the dataset used in the paper from [here](https://vered1986.github.io/papers/Tratz2011_Dataset.tar.gz).
2. You'll need a collection of word embeddings: the path-based model uses the
   word embeddings as part of the path representation, and the distributional
   models use the word embeddings directly as prediction features.
3. The path-based model requires a collection of syntactic dependency parses
   that connect the constituents for each noun compound.

At the moment, this repository does not contain the tools for generating this
data, but we will provide references to existing datasets and plan to add tools
to generate the data in the future.

# Contents

The following source code is included here:

* `learn_path_embeddings.py` is a script that trains and evaluates a path-based
  model to predict a noun-compound relationship given labeled noun-compounds and
  dependency parse paths.
* `learn_classifier.py` is a script that trains and evaluates a classifier based
  on any combination of paths, word embeddings, and noun-compound embeddings.
* `get_indicative_paths.py` is a script that generates the most indicative
  syntactic dependency paths for a particular relationship.

# Dependencies

* [TensorFlow](http://www.tensorflow.org/): see detailed installation
  instructions at that site.
* [SciKit Learn](http://scikit-learn.org/): you can probably just install this
  with `pip install sklearn`.

# Creating the Model

This section describes the necessary steps that you must follow to reproduce the
results in the paper.

## Generate/Download Path Data

TBD! Our plan is to make the aggregate path data available that was used to
train path embeddings and classifiers; however, this will be released
separately.

## Generate/Download Embedding Data

TBD! While we used the standard Glove vectors for the relata embeddings, the NC
embeddings were generated separately. Our plan is to make that data available,
but it will be released separately.

## Create Path Embeddings

Create the path embeddings using `learn_path_embeddings.py`.  This shell script
fragment will iterate through each dataset, split, and corpus to generate path
embeddings for each.

    for DATASET in tratz/fine_grained tratz/coarse_grained ; do
      for SPLIT in random lexical_head lexical_mod lexical_full ; do
        for CORPUS in wiki_gigiawords ; do
          python learn_path_embeddings.py \
            --dataset_dir ~/lexnet/datasets \
            --dataset "${DATASET}" \
            --corpus "${SPLIT}/${CORPUS}" \
            --embeddings_base_path ~/lexnet/embeddings \
            --logdir /tmp/learn_path_embeddings
        done
      done
    done

The path embeddings will be placed in the directory specified by
`--embeddings_base_path`.

## Train classifiers

Train classifiers and evaluate on the validation and test data using
`train_classifiers.py` script.  This shell script fragment will iterate through
each dataset, split, corpus, and model type to train and evaluate classifiers.

    LOGDIR=/tmp/learn_classifier
    for DATASET in tratz/fine_grained tratz/coarse_grained ; do
      for SPLIT in random lexical_head lexical_mod lexical_full ; do
        for CORPUS in wiki_gigiawords ; do
          for MODEL in dist dist-nc path integrated integrated-nc ; do
            # Filename for the log that will contain the classifier results.
            LOGFILE=$(echo "${DATASET}.${SPLIT}.${CORPUS}.${MODEL}.log" | sed -e "s,/,.,g")
            python learn_classifier.py \
              --dataset_dir ~/lexnet/datasets \
              --dataset "${DATASET}" \
              --corpus "${SPLIT}/${CORPUS}" \
              --embeddings_base_path ~/lexnet/embeddings \
              --logdir ${LOGDIR} \
              --input "${MODEL}" > "${LOGDIR}/${LOGFILE}"
          done
        done
      done
    done

The log file will contain the final performance (precision, recall, F1) on the
train, dev, and test sets, and will include a confusion matrix for each.

# Contact

If you have any questions, issues, or suggestions, feel free to contact either
@vered1986 or @waterson.

If you use this code for any published research, please include the following citation:

Olive Oil Is Made of Olives, Baby Oil Is Made for Babies: Interpreting Noun Compounds Using Paraphrases in a Neural Model. 
Vered Shwartz and Chris Waterson. NAACL 2018. [link](https://arxiv.org/pdf/1803.08073.pdf).
