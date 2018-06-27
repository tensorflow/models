# Seq2Species: Neural Network Models for Species Classification

*A deep learning solution for read-level taxonomic classification with 16s.*

Recent improvements in sequencing technology have made possible large, public
databases of biological sequencing data, bringing about new data richness for
many important problems in bioinformatics. However, this growing availability of
data creates a need for analysis methods capable of efficiently handling these
large sequencing datasets. We on the [Genomics team in Google
Brain](https://ai.google/research/teams/brain/healthcare-biosciences) are
particularly interested in the class of problems which can be framed as
assigning meaningful labels to short biological sequences, and are exploring the
possiblity of creating a general deep learning solution for solving this class
of sequence-labeling problems. We are excited to share our initial progress in
this direction by releasing Seq2Species, an open-source neural network framework
for [TensorFlow](https://www.tensorflow.org/) for predicting read-level
taxonomic labels from genomic sequence. Our release includes all the code
necessary to train new Seq2Species models.

## About Seq2Species

Briefly, Seq2Species provides a framework for training deep neural networks to
predict database-derived labels directly from short reads of DNA. Thus far, our
research has focused predominantly on demonstrating the value of this deep
learning approach on the problem of determining the species of origin of
next-generation sequencing reads from [16S ribosomal
DNA](https://en.wikipedia.org/wiki/16S_ribosomal_RNA). We used this
Seq2Species framework to train depthwise separable convolutional neural networks
on short subsequences from the 16S genes of more than 13 thousand distinct
species. The resulting classification model assign species-level probabilities
to individual 16S reads.

For more information about the use cases we have explored, or for technical
details describing how Seq2Species work, please see our
[preprint](https://www.biorxiv.org/content/early/2018/06/22/353474).

## Installation

Training Seq2Species models requires installing the following dependencies:

* python 2.7

* protocol buffers

* numpy

* absl

### Dependencies

Detailed instructions for installing TensorFlow are available on the [Installing
TensorFlow](https://www.tensorflow.org/install/) website. Please follow the
full instructions for installing TensorFlow with GPU support. For most
users, the following command will suffice for continuing with CPU support only:
```bash
# For CPU
pip install --upgrade tensorflow
```

The TensorFlow installation should also include installation of the numpy and
absl libraries, which are two of TensorFlow's python dependencies. If
necessary, instructions for standalone installation are available:

* [numpy](https://scipy.org/install.html)

* [absl](https://github.com/abseil/abseil-py)

Information about protocol buffers, as well as download and installation
intructions for the protocol buffer (protobuf) compiler, are available on the [Google
Developers website](https://developers.google.com/protocol-buffers/). A typical
Ubuntu user can install this library using `apt-get`:
```bash
sudo apt-get install protobuf-compiler
```

### Clone

Now, clone `tensorflow/models` to start working with the code:
```bash
git clone https://github.com/tensorflow/models.git
```

### Protobuf Compilation

Seq2Species uses protobufs to store and save dataset and model metadata. Before
the framework can be used to build and train models, the protobuf libraries must
be compiled. This can be accomplished using the following command:
```bash
# From tensorflow/models/research
protoc seq2species/protos/seq2label.proto --python_out=.
```

### Testing the Installation

One can test that Seq2Species has been installed correctly by running the
following command:
```bash
python seq2species/run_training_test.py
```

## Usage Information

Input data to Seq2Species models should be [tf.train.Example protocol messages](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/example/example.proto) stored in
[TFRecord format](https://www.tensorflow.org/versions/r1.0/api_guides/python/python_io#tfrecords_format_details).
Specifically, the input pipeline expects tf.train.Examples with a 'sequence' field
containing a genomic sequence as an upper-case string, as one field for each
target label (e.g. 'species'). There should also be an accompanying
Seq2LabelDatasetInfo text protobuf containing metadata about the input, including
the possible label values for each target.

Below, we give an example command that could be used to launch training for 1000
steps, assuming that appropriate data and metadata files are stored at
`${TFRECORD}` and `${DATASET_INFO}`:
```bash
python seq2species/run_training.py --train_files ${TFRECORD}
--metadata_path ${DATASET_INFO} --hparams 'train_steps=1000'
--logdir $HOME/seq2species
```
This will output [TensorBoard
summaries](https://www.tensorflow.org/guide/summaries_and_tensorboard), [TensorFlow
checkpoints](https://www.tensorflow.org/guide/variables#checkpoint_files), Seq2LabelModelInfo and
Seq2LabelExperimentMeasures metadata to the logdir `$HOME/seq2species`.

### Preprocessed Seq2Species Data

We have provided preprocessed data based on 16S reference sequences from the
[NCBI RefSeq Targeted Loci
Project](https://www.ncbi.nlm.nih.gov/refseq/targetedloci/) in a Seq2Species
bucket on Google Cloud Storage. After installing the
[Cloud SDK](https://cloud.google.com/sdk/install),
one can download those data (roughly 25 GB) to a local directory `${DEST}` using
the `gsutil` command:
```bash
BUCKET=gs://brain-genomics-public/research/seq2species
mkdir -p ${DEST}
gsutil -m cp ${BUCKET}/* ${DEST}
```

To check if the copy has completed successsfully, check the `${DEST}` directory:
```bash
ls -1 ${DEST}
```
which should produce:
```bash
ncbi_100bp_revcomp.dataset_info.pbtxt
ncbi_100bp_revcomp.tfrecord
```

The following command can be used to train a copy of one of our best-perfoming
deep neural network models for 100 base pair (bp) data. This command also
illustrates how to set hyperparameter values explicitly from the commandline.
The file `configuration.py` provides a full list of hyperparameters, their descriptions,
and their default values. Additional flags are described at the top of
`run_training.py`.
```bash
python seq2species/run_training.py \
--num_filters 3 \
--noise_rate 0.04 \
--train_files ${DEST}/ncbi_100bp_revcomp.tfrecord \
--metadata_path ${DEST}/ncbi_100bp_revcomp.dataset_info.pbtxt \
--logdir $HOME/seq2species \
--hparams 'filter_depths=[1,1,1],filter_widths=[5,9,13],grad_clip_norm=20.0,keep_prob=0.94017831318,
lr_decay=0.0655052811,lr_init=0.000469689635793,lrelu_slope=0.0125376069918,min_read_length=100,num_fc_layers=2,num_fc_units=2828,optimizer=adam,optimizer_hp=0.885769367218,pointwise_depths=[84,58,180],pooling_type=avg,train_steps=3000000,use_depthwise_separable=true,weight_scale=1.18409526348'
```

### Visualization

[TensorBoard](https://github.com/tensorflow/tensorboard) can be used to
visualize training curves and other metrics stored in the summary files produced
by `run_training.py`. Use the following command to launch a TensorBoard instance
for the example model directory `$HOME/seq2species`:
```bash
tensorboard --logdir=$HOME/seq2species
```

## Contact

Any issues with the Seq2Species framework should be filed with the
[TensorFlow/models issue tracker](https://github.com/tensorflow/models/issues).
Questions regarding Seq2Species capabilities can be directed to
[seq2species-interest@google.com](mailto:seq2species-interest@google.com). This
code is maintained by [@apbusia](https://github.com/apbusia) and
[@depristo](https://github.com/depristo).
