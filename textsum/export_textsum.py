"""

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, traceback
import tensorflow as tf
import batch_reader
import data
from tensorflow.contrib.session_bundle import exporter
import seq2seq_attention_decode
import seq2seq_attention_model

tf.app.flags.DEFINE_string("export_dir", "exports/textsum",
                           "Directory where to export textsum model.")

tf.app.flags.DEFINE_string('checkpoint_dir', 'log_root',
                            "Directory where to read training checkpoints.")
tf.app.flags.DEFINE_integer('export_version', 1, 'version number of the model.')
tf.app.flags.DEFINE_bool("use_checkpoint_v2", False,
                     "If true, write v2 checkpoint files.")
tf.app.flags.DEFINE_integer('random_seed', 111, 'A seed value for randomness.')
tf.app.flags.DEFINE_string('vocab_path',
                           '/media/daniel/Data/dataForTraining/vocab', 'Path expression to text vocabulary file.')
tf.app.flags.DEFINE_string('data_path',
                           '/media/daniel/Data/dataForTraining/srcArticlesTrainBinary/data-*', 'Path expression to tf.Example.')
tf.app.flags.DEFINE_string('article_key', 'article',
                           'tf.Example feature key for article.')
tf.app.flags.DEFINE_string('abstract_key', 'headline',
                           'tf.Example feature key for abstract.')
tf.app.flags.DEFINE_integer('max_article_sentences', 2,
                            'Max number of first sentences to use from the '
                            'article')
tf.app.flags.DEFINE_integer('max_abstract_sentences', 100,
                            'Max number of first sentences to use from the '
                            'abstract')
tf.app.flags.DEFINE_bool('use_bucketing', False,
                         'Whether bucket articles of similar length.')
tf.app.flags.DEFINE_bool('truncate_input', True,
                         'Truncate inputs that are too long. If False, '
                         'examples that are too long are discarded.')
tf.app.flags.DEFINE_integer('num_gpus', 0, 'Number of gpus used.')


FLAGS = tf.app.flags.FLAGS

def Export():
    try:
        with tf.Graph().as_default():
            vocab = data.Vocab(FLAGS.vocab_path, 1000000)
            # Check for presence of required special tokens.
            assert vocab.WordToId(data.PAD_TOKEN) > 0
            assert vocab.WordToId(data.UNKNOWN_TOKEN) >= 0
            assert vocab.WordToId(data.SENTENCE_START) > 0
            assert vocab.WordToId(data.SENTENCE_END) > 0

            batch_size = 8

            hps = seq2seq_attention_model.HParams(
                mode='decode', #FLAGS.mode,  # train, eval, decode
                min_lr=0.01,  # min learning rate.
                lr=0.15,  # learning rate
                batch_size=batch_size,
                enc_layers=4,
                enc_timesteps=120,
                dec_timesteps=30,
                min_input_len=2,  # discard articles/summaries < than this
                num_hidden=256,  # for rnn cell
                emb_dim=128,  # If 0, don't use embedding
                max_grad_norm=2,
                num_softmax_samples=4096)  # If 0, no sampled softmax.

            batcher = batch_reader.Batcher(
                FLAGS.data_path, vocab, hps, FLAGS.article_key,
                FLAGS.abstract_key, FLAGS.max_article_sentences,
                FLAGS.max_abstract_sentences, bucketing=FLAGS.use_bucketing,
                truncate_input=FLAGS.truncate_input)
            tf.set_random_seed(FLAGS.random_seed)

            decode_mdl_hps = hps
            # Only need to restore the 1st step and reuse it since
            # we keep and feed in state for each step's output.
            decode_mdl_hps = hps._replace(dec_timesteps=1)
            model = seq2seq_attention_model.Seq2SeqAttentionModel(
                decode_mdl_hps, vocab, num_gpus=FLAGS.num_gpus)
            decoder = seq2seq_attention_decode.BSDecoder(model, batcher, hps, vocab)
            #decoder.DecodeLoop()

            serialized_tf_example = tf.placeholder(tf.string, name='tf_example')
            feature_configs = {
                'image/encoded': tf.FixedLenFeature(shape=[], dtype=tf.string),
            }
            tf_example = tf.parse_example(serialized_tf_example, feature_configs)
            #model.build_graph()
            saver = tf.train.Saver(sharded=True)
            with tf.Session() as sess:
                
                # Restore variables from training checkpoints.
                ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    print('Successfully loaded model from %s at step=%s.' %
                        (ckpt.model_checkpoint_path, global_step))
                else:
                    print('No checkpoint file found at %s' % FLAGS.checkpoint_dir)
                    return

                # Export model
                print('Exporting trained model to %s' % FLAGS.export_dir)
                init_op = tf.group(tf.initialize_all_tables(), name='init_op')
                model_exporter = exporter.Exporter(saver)

                classification_signature = exporter.classification_signature(
                    input_tensor=serialized_tf_example,
                    classes_tensor=classes,
                    scores_tensor=values)
                    
                named_graph_signature = {
                    'inputs': exporter.generic_signature({'images': jpegs}),
                    'outputs': exporter.generic_signature({
                        'classes': classes,
                        'scores': values
                    })}

                model_exporter.init(
                    init_op=init_op,
                    default_graph_signature=classification_signature,
                    named_graph_signatures=named_graph_signature)

                model_exporter.export(FLAGS.export_dir, tf.constant(global_step), sess)
                print('Successfully exported model to %s' % FLAGS.export_dir)
    except:
        traceback.print_exc()
        pass


def main(_):
    Export()

if __name__ == "__main__":
    tf.app.run()