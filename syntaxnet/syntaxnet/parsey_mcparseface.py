import os
import shutil

import tensorflow as tf

from tensorflow.python.platform import tf_logging as logging
from syntaxnet import parser_eval
from syntaxnet.ops import gen_parser_ops
from syntaxnet import structured_graph_builder
from tensorflow_serving.session_bundle import exporter

def Build(sess, document_source, FLAGS):
  """Builds a sub-network, which will be either the tagger or the parser

  Args:
    sess: tensorflow session to use
    document_source: the input of serialized document objects to process

  Flags: (taken from FLAGS argument)
    num_actions: number of possible golden actions
    feature_sizes: size of each feature vector
    domain_sizes: number of possible feature ids in each feature vector
    embedding_dims: embedding dimension for each feature group
    
    hidden_layer_sizes: Comma separated list of hidden layer sizes.
    arg_prefix: Prefix for context parameters.
    beam_size: Number of slots for beam parsing.
    max_steps: Max number of steps to take.
    task_context: Path to a task context with inputs and parameters for feature extractors.
    input: Name of the context input to read data from.
    output: Name of the context input to write data to.
    graph_builder: 'greedy' or 'structured'
    batch_size: Number of sentences to process in parallel.
    slim_model: Whether to expect only averaged variables.
    model_path: Path to model parameters.

  Return:
    returns the tensor which will contain the serialized document objects.

  """
  task_context = FLAGS["task_context"]
  arg_prefix = FLAGS["arg_prefix"]
  num_actions = FLAGS["num_actions"]
  feature_sizes = FLAGS["feature_sizes"]
  domain_sizes = FLAGS["domain_sizes"]
  embedding_dims = FLAGS["embedding_dims"]
  hidden_layer_sizes = map(int, FLAGS["hidden_layer_sizes"].split(','))
  beam_size = FLAGS["beam_size"]
  max_steps = FLAGS["max_steps"]
  batch_size = FLAGS["batch_size"]
  corpus_name = FLAGS["input"]
  slim_model = FLAGS["slim_model"]
  model_path = FLAGS["model_path"]
  
  parser = structured_graph_builder.StructuredGraphBuilder(
        num_actions,
        feature_sizes,
        domain_sizes,
        embedding_dims,
        hidden_layer_sizes,
        gate_gradients=True,
        arg_prefix=arg_prefix,
        beam_size=beam_size,
        max_steps=max_steps)
  
  parser.AddEvaluation(task_context,
                       batch_size,
                       corpus_name=corpus_name,
                       evaluation_max_steps=max_steps,
   		       document_source=document_source)

  parser.AddSaver(slim_model)
  sess.run(parser.inits.values())
  parser.saver.restore(sess, model_path)
  
  return parser.evaluation['documents']

def GetFeatureSize(task_context, arg_prefix):
  with tf.variable_scope("fs_"+arg_prefix):
    with tf.Session() as sess:
      return sess.run(gen_parser_ops.feature_size(task_context=task_context,
                      arg_prefix=arg_prefix))

def main(unused_argv):
  logging.set_verbosity(logging.INFO)

  model_dir="syntaxnet/models/parsey_mcparseface"
  task_context="%s/context.pbtxt" % model_dir

  common_params = {
      "task_context":  task_context,
      "beam_size":     8,
      "max_steps":     1000,
      "output":        "stdout-conll",
      "graph_builder": "structured",
      "batch_size":    1024,
      "slim_model":    True,
      }
        
  model = {
      	"brain_tagger": {
            "arg_prefix":         "brain_tagger",
            "hidden_layer_sizes": "64",
            "input":              "stdin",
            "model_path":         "%s/tagger-params" % model_dir,

            },
        "brain_parser": {
            "arg_prefix":         "brain_parser",
            "hidden_layer_sizes": "512,512",
            "input":              "stdin-conll",
            "model_path":         "%s/parser-params" % model_dir,
            },
      }
  
  for prefix in ["brain_tagger","brain_parser"]:
      model[prefix].update(common_params)
      feature_sizes, domain_sizes, embedding_dims, num_actions = GetFeatureSize(task_context, prefix)
      model[prefix].update({'feature_sizes': feature_sizes,
                               'domain_sizes': domain_sizes,
                               'embedding_dims': embedding_dims,
                               'num_actions': num_actions })

  with tf.Session() as sess:
      unused_text_input = tf.constant(["parsey is the greatest"], tf.string)
      document_source = gen_parser_ops.document_source(text=unused_text_input,
                                                       task_context=task_context,
                                                       corpus_name="stdin",
                                                       batch_size=common_params['batch_size'],
					               documents_from_input=True)

      for prefix in ["brain_tagger","brain_parser"]:
          with tf.variable_scope(prefix):
              if True or prefix == "brain_tagger":
                  source = document_source.documents if prefix == "brain_tagger" else model["brain_tagger"]["documents"]
                  model[prefix]["documents"] = Build(sess, source, model[prefix])


      sink = gen_parser_ops.document_sink(model["brain_parser"]["documents"],
                                      task_context=task_context,
                                      corpus_name="stdout-conll")
      ExportModel(sess)
      sess.run(sink)

def ExportModel(sess):
  # save the model in various ways. this erases any previously saved model
  model_dir = '/tmp/model/parsey_mcparseface'
  if os.path.isdir(model_dir):
    shutil.rmtree(model_dir);

  ## using TF Serving exporter to load into a TF Serving session bundle
  logging.info('Exporting trained model to %s', model_dir)
  saver = tf.train.Saver()
  model_exporter = exporter.Exporter(saver)
  signature = exporter.generic_signature({})
  model_exporter.init(sess.graph.as_graph_def(),
                      default_graph_signature=signature)
  model_exporter.export(model_dir, tf.constant(1), sess)

  ## using a SummaryWriter so graph can be loaded in TensorBoard
  writer = tf.train.SummaryWriter(model_dir, sess.graph)
  writer.flush()

  ## exporting the graph as a text protobuf, to view graph manualy
  f1 = open(model_dir + '/graph.pbtxt', 'w+');
  print >>f1, str(tf.get_default_graph().as_graph_def())


if __name__ == '__main__':
  tf.app.run()
