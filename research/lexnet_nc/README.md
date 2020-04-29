![No Maintenance Intended](https://img.shields.io/badge/No%20Maintenance%20Intended-%E2%9C%95-red.svg)
![TensorFlow Requirement: 1.x](https://img.shields.io/badge/TensorFlow%20Requirement-1.x-brightgreen)
![TensorFlow 2 Not Supported](https://img.shields.io/badge/TensorFlow%202%20Not%20Supported-%E2%9C%95-red.svg)

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
   *purpose*), and generally may consist of tens of classes.  You can download
   the dataset used in the paper from
   [here](https://vered1986.github.io/papers/Tratz2011_Dataset.tar.gz).
2. A collection of word embeddings: the path-based model uses the word
   embeddings as part of the path representation, and the distributional models
   use the word embeddings directly as prediction features.
3. The path-based model requires a collection of syntactic dependency parses
   that connect the constituents for each noun compound. To generate these,
   you'll need a corpus from which to train this data; we used Wikipedia and the
   [LDC GigaWord5](https://catalog.ldc.upenn.edu/LDC2011T07) corpora.

# Contents

The following source code is included here:

* `learn_path_embeddings.py` is a script that trains and evaluates a path-based
  model to predict a noun-compound relationship given labeled noun-compounds and
  dependency parse paths.
* `learn_classifier.py` is a script that trains and evaluates a classifier based
  on any combination of paths, word embeddings, and noun-compound embeddings.
* `get_indicative_paths.py` is a script that generates the most indicative
  syntactic dependency paths for a particular relationship.

Also included are utilities for preparing data for training:

* `text_embeddings_to_binary.py` converts a text file containing word embeddings
  into a binary file that is quicker to load.
* `extract_paths.py` finds all the dependency paths that connect words in a
  corpus.
* `sorted_paths_to_examples.py` processes the output of `extract_paths.py` to
  produce summarized training data.

This code (in particular, the utilities used to prepare the data) differs from
the code that was used to prepare data for the paper. Notably, we used a
proprietary dependency parser instead of spaCy, which is used here.

# Dependencies

* [TensorFlow](http://www.tensorflow.org/): see detailed installation
  instructions at that site.
* [SciKit Learn](http://scikit-learn.org/): you can probably just install this
  with `pip install sklearn`.
* [SpaCy](https://spacy.io/): `pip install spacy` ought to do the trick, along
  with the English model.

# Creating the Model

This sections described the steps necessary to create and evaluate the model
described in the paper.

## Generate Path Data

To begin, you need three text files:

1. **Corpus**. This file should contain natural language sentences, written with
   one sentence per line.  For purposes of exposition, we'll assume that you
   have English Wikipedia serialized this way in `${HOME}/data/wiki.txt`.
2. **Labeled Noun Compound Pairs**.  This file contain (modfier, head, label)
   tuples, tab-separated, with one per line.  The *label* represented the
   relationship between the head and the modifier; e.g., if `purpose` is one
   your labels, you could possibly include `tooth<tab>paste<tab>purpose`.
3. **Word Embeddings**. We used the
   [GloVe](https://nlp.stanford.edu/projects/glove/) word embeddings; in
   particular the 6B token, 300d variant.  We'll assume you have this file as
   `${HOME}/data/glove.6B.300d.txt`.

We first processed the embeddings from their text format into something that we
can load a little bit more quickly:

    ./text_embeddings_to_binary.py \
      --input ${HOME}/data/glove.6B.300d.txt \
      --output_vocab ${HOME}/data/vocab.txt \
      --output_npy ${HOME}/data/glove.6B.300d.npy

Next, we'll extract all the dependency parse paths connecting our labeled pairs
from the corpus.  This process takes a *looooong* time, but is trivially
parallelized using map-reduce if you have access to that technology.

    ./extract_paths.py \
      --corpus ${HOME}/data/wiki.txt \
      --labeled_pairs ${HOME}/data/labeled-pairs.tsv \
      --output ${HOME}/data/paths.tsv

The file it produces (`paths.tsv`) is a tab-separated file that contains the
modifier, the head, the label, the encoded path, and the sentence from which the
path was drawn.  (This last is mostly for sanity checking.)  A sample row might
look something like this (where newlines would actually be tab characters):

    navy
    captain
    owner_emp_use
    <X>/PROPN/dobj/>::enter/VERB/ROOT/^::follow/VERB/advcl/<::in/ADP/prep/<::footstep/NOUN/pobj/<::of/ADP/prep/<::father/NOUN/pobj/<::bover/PROPN/appos/<::<Y>/PROPN/compound/<
    He entered the Royal Navy following in the footsteps of his father Captain John Bover and two of his elder brothers as volunteer aboard HMS Perseus

This file must be sorted as follows:

    sort -k1,3 -t$'\t' paths.tsv > sorted.paths.tsv

In particular, rows with the same modifier, head, and label must appear
contiguously.

We next create a file that contains all the relation labels from our original
labeled pairs:

    awk 'BEGIN {FS="\t"} {print $3}' < ${HOME}/data/labeled-pairs.tsv \
      | sort -u > ${HOME}/data/relations.txt

With these in hand, we're ready to produce the train, validation, and test data:

    ./sorted_paths_to_examples.py \
       --input ${HOME}/data/sorted.paths.tsv \
       --vocab ${HOME}/data/vocab.txt \
       --relations ${HOME}/data/relations.txt \
       --splits ${HOME}/data/splits.txt \
       --output_dir ${HOME}/data

Here, `splits.txt` is a file that indicates which "split" (train, test, or
validation) you want the pair to appear in.  It should be a tab-separate file
which conatins the modifier, head, and the dataset ( `train`, `test`, or `val`)
into which the pair should be placed; e.g.,:

    tooth <TAB> paste <TAB> train
    banana <TAB> seat <TAB> test

The program will produce a separate file for each dataset split in the directory
specified by `--output_dir`.  Each file is contains `tf.train.Example` protocol
buffers encoded using the `TFRecord` file format.

## Create Path Embeddings

Now we're ready to train the path embeddings using `learn_path_embeddings.py`:

    ./learn_path_embeddings.py \
        --train ${HOME}/data/train.tfrecs.gz \
        --val ${HOME}/data/val.tfrecs.gz \
        --text ${HOME}/data/test.tfrecs.gz \
        --embeddings ${HOME}/data/glove.6B.300d.npy
        --relations ${HOME}/data/relations.txt
        --output ${HOME}/data/path-embeddings \
        --logdir /tmp/learn_path_embeddings

The path embeddings will be placed at the location specified by `--output`.

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
