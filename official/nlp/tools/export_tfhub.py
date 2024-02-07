# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Exports a BERT-like encoder and its preprocessing as SavedModels for TF Hub.

This tool creates preprocessor and encoder SavedModels suitable for uploading
to https://tfhub.dev that implement the preprocessor and encoder APIs defined
at https://www.tensorflow.org/hub/common_saved_model_apis/text.

For a full usage guide, see
https://github.com/tensorflow/models/blob/master/official/nlp/docs/tfhub.md

Minimal usage examples:

1) Exporting an Encoder from checkpoint and config.

```
export_tfhub \
  --encoder_config_file=${BERT_DIR:?}/bert_encoder.yaml \
  --model_checkpoint_path=${BERT_DIR:?}/bert_model.ckpt \
  --vocab_file=${BERT_DIR:?}/vocab.txt \
  --export_type=model \
  --export_path=/tmp/bert_model
```

An --encoder_config_file can specify encoder types other than BERT.
For BERT, a --bert_config_file in the legacy JSON format can be passed instead.

Flag --vocab_file (and flag --do_lower_case, whose default value is guessed
from the vocab_file path) capture how BertTokenizer was used in pre-training.
Use flag --sp_model_file instead if SentencepieceTokenizer was used.

Changing --export_type to model_with_mlm additionally creates an `.mlm`
subobject on the exported SavedModel that can be called to produce
the logits of the Masked Language Model task from pretraining.
The help string for flag --model_checkpoint_path explains the checkpoint
formats required for each --export_type.


2) Exporting a preprocessor SavedModel

```
export_tfhub \
  --vocab_file ${BERT_DIR:?}/vocab.txt \
  --export_type preprocessing --export_path /tmp/bert_preprocessing
```

Be sure to use flag values that match the encoder and how it has been
pre-trained (see above for --vocab_file vs --sp_model_file).

If your encoder has been trained with text preprocessing for which tfhub.dev
already has SavedModel, you could guide your users to reuse that one instead
of exporting and publishing your own.

TODO(b/175369555): When exporting to users of TensorFlow 2.4, add flag
`--experimental_disable_assert_in_preprocessing`.
"""

from absl import app
from absl import flags
import gin

from official.legacy.bert import configs
from official.modeling import hyperparams
from official.nlp.configs import encoders
from official.nlp.tools import export_tfhub_lib

FLAGS = flags.FLAGS

flags.DEFINE_enum(
    "export_type", "model",
    ["model", "model_with_mlm", "preprocessing"],
    "The overall type of SavedModel to export. Flags "
    "--bert_config_file/--encoder_config_file and --vocab_file/--sp_model_file "
    "control which particular encoder model and preprocessing are exported.")
flags.DEFINE_string(
    "export_path", None,
    "Directory to which the SavedModel is written.")
flags.DEFINE_string(
    "encoder_config_file", None,
    "A yaml file representing `encoders.EncoderConfig` to define the encoder "
    "(BERT or other). "
    "Exactly one of --bert_config_file and --encoder_config_file can be set. "
    "Needed for --export_type model and model_with_mlm.")
flags.DEFINE_string(
    "bert_config_file", None,
    "A JSON file with a legacy BERT configuration to define the BERT encoder. "
    "Exactly one of --bert_config_file and --encoder_config_file can be set. "
    "Needed for --export_type model and model_with_mlm.")
flags.DEFINE_bool(
    "copy_pooler_dense_to_encoder", False,
    "When the model is trained using `BertPretrainerV2`, the pool layer "
    "of next sentence prediction task exists in `ClassificationHead` passed "
    "to `BertPretrainerV2`. If True, we will copy this pooler's dense layer "
    "to the encoder that is exported by this tool (as in classic BERT). "
    "Using `BertPretrainerV2` and leaving this False exports an untrained "
    "(randomly initialized) pooling layer, which some authors recommend for "
    "subsequent fine-tuning,")
flags.DEFINE_string(
    "model_checkpoint_path", None,
    "File path to a pre-trained model checkpoint. "
    "For --export_type model, this has to be an object-based (TF2) checkpoint "
    "that can be restored to `tf.train.Checkpoint(encoder=encoder)` "
    "for the `encoder` defined by the config file."
    "(Legacy checkpoints with `model=` instead of `encoder=` are also "
    "supported for now.) "
    "For --export_type model_with_mlm, it must be restorable to "
    "`tf.train.Checkpoint(**BertPretrainerV2(...).checkpoint_items)`. "
    "(For now, `tf.train.Checkpoint(pretrainer=BertPretrainerV2(...))` is also "
    "accepted.)")
flags.DEFINE_string(
    "vocab_file", None,
    "For encoders trained on BertTokenzier input: "
    "the vocabulary file that the encoder model was trained with. "
    "Exactly one of --vocab_file and --sp_model_file can be set. "
    "Needed for --export_type model, model_with_mlm and preprocessing.")
flags.DEFINE_string(
    "sp_model_file", None,
    "For encoders trained on SentencepieceTokenzier input: "
    "the SentencePiece .model file that the encoder model was trained with. "
    "Exactly one of --vocab_file and --sp_model_file can be set. "
    "Needed for --export_type model, model_with_mlm and preprocessing.")
flags.DEFINE_bool(
    "do_lower_case", None,
    "Whether to lowercase before tokenization. "
    "If left as None, and --vocab_file is set, do_lower_case will be enabled "
    "if 'uncased' appears in the name of --vocab_file. "
    "If left as None, and --sp_model_file set, do_lower_case defaults to true. "
    "Needed for --export_type model, model_with_mlm and preprocessing.")
flags.DEFINE_integer(
    "default_seq_length", 128,
    "The sequence length of preprocessing results from "
    "top-level preprocess method. This is also the default "
    "sequence length for the bert_pack_inputs subobject."
    "Needed for --export_type preprocessing.")
flags.DEFINE_bool(
    "tokenize_with_offsets", False,  # TODO(b/181866850)
    "Whether to export a .tokenize_with_offsets subobject for "
    "--export_type preprocessing.")
flags.DEFINE_multi_string(
    "gin_file", default=None,
    help="List of paths to the config files.")
flags.DEFINE_multi_string(
    "gin_params", default=None,
    help="List of Gin bindings.")
flags.DEFINE_bool(  # TODO(b/175369555): Remove this flag and its use.
    "experimental_disable_assert_in_preprocessing", False,
    "Export a preprocessing model without tf.Assert ops. "
    "Usually, that would be a bad idea, except TF2.4 has an issue with "
    "Assert ops in tf.functions used in Dataset.map() on a TPU worker, "
    "and omitting the Assert ops lets SavedModels avoid the issue.")


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_params)

  if bool(FLAGS.vocab_file) == bool(FLAGS.sp_model_file):
    raise ValueError("Exactly one of `vocab_file` and `sp_model_file` "
                     "can be specified, but got %s and %s." %
                     (FLAGS.vocab_file, FLAGS.sp_model_file))
  do_lower_case = export_tfhub_lib.get_do_lower_case(
      FLAGS.do_lower_case, FLAGS.vocab_file, FLAGS.sp_model_file)

  if FLAGS.export_type in ("model", "model_with_mlm"):
    if bool(FLAGS.bert_config_file) == bool(FLAGS.encoder_config_file):
      raise ValueError("Exactly one of `bert_config_file` and "
                       "`encoder_config_file` can be specified, but got "
                       "%s and %s." %
                       (FLAGS.bert_config_file, FLAGS.encoder_config_file))
    if FLAGS.bert_config_file:
      bert_config = configs.BertConfig.from_json_file(FLAGS.bert_config_file)
      encoder_config = None
    else:
      bert_config = None
      encoder_config = encoders.EncoderConfig()
      encoder_config = hyperparams.override_params_dict(
          encoder_config, FLAGS.encoder_config_file, is_strict=True)
    export_tfhub_lib.export_model(
        FLAGS.export_path,
        bert_config=bert_config,
        encoder_config=encoder_config,
        model_checkpoint_path=FLAGS.model_checkpoint_path,
        vocab_file=FLAGS.vocab_file,
        sp_model_file=FLAGS.sp_model_file,
        do_lower_case=do_lower_case,
        with_mlm=FLAGS.export_type == "model_with_mlm",
        copy_pooler_dense_to_encoder=FLAGS.copy_pooler_dense_to_encoder)

  elif FLAGS.export_type == "preprocessing":
    export_tfhub_lib.export_preprocessing(
        FLAGS.export_path,
        vocab_file=FLAGS.vocab_file,
        sp_model_file=FLAGS.sp_model_file,
        do_lower_case=do_lower_case,
        default_seq_length=FLAGS.default_seq_length,
        tokenize_with_offsets=FLAGS.tokenize_with_offsets,
        experimental_disable_assert=
        FLAGS.experimental_disable_assert_in_preprocessing)

  else:
    raise app.UsageError(
        "Unknown value '%s' for flag --export_type" % FLAGS.export_type)


if __name__ == "__main__":
  app.run(main)
