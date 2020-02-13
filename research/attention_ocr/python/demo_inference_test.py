#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os
import demo_inference
import tensorflow as tf
from tensorflow.python.training import monitored_session

_CHECKPOINT = 'model.ckpt-399731'
_CHECKPOINT_URL = 'http://download.tensorflow.org/models/attention_ocr_2017_08_09.tar.gz'


class DemoInferenceTest(tf.test.TestCase):
  def setUp(self):
    super(DemoInferenceTest, self).setUp()
    for suffix in ['.meta', '.index', '.data-00000-of-00001']:
      filename = _CHECKPOINT + suffix
      self.assertTrue(tf.gfile.Exists(filename),
                      msg='Missing checkpoint file %s. '
                          'Please download and extract it from %s' %
                          (filename, _CHECKPOINT_URL))
    self._batch_size = 32
    tf.flags.FLAGS.dataset_dir = os.path.join(os.path.dirname(__file__), 'datasets/testdata/fsns')

  def test_moving_variables_properly_loaded_from_a_checkpoint(self):
    batch_size = 32
    dataset_name = 'fsns'
    images_placeholder, endpoints = demo_inference.create_model(batch_size,
                                                                dataset_name)
    image_path_pattern = 'testdata/fsns_train_%02d.png'
    images_data = demo_inference.load_images(image_path_pattern, batch_size,
                                             dataset_name)
    tensor_name = 'AttentionOcr_v1/conv_tower_fn/INCE/InceptionV3/Conv2d_2a_3x3/BatchNorm/moving_mean'
    moving_mean_tf = tf.get_default_graph().get_tensor_by_name(
      tensor_name + ':0')
    reader = tf.train.NewCheckpointReader(_CHECKPOINT)
    moving_mean_expected = reader.get_tensor(tensor_name)

    session_creator = monitored_session.ChiefSessionCreator(
      checkpoint_filename_with_path=_CHECKPOINT)
    with monitored_session.MonitoredSession(
        session_creator=session_creator) as sess:
      moving_mean_np = sess.run(moving_mean_tf,
                                feed_dict={images_placeholder: images_data})

    self.assertAllEqual(moving_mean_expected, moving_mean_np)

  def test_correct_results_on_test_data(self):
    image_path_pattern = 'testdata/fsns_train_%02d.png'
    predictions = demo_inference.run(_CHECKPOINT, self._batch_size,
                                     'fsns',
                                     image_path_pattern)
    self.assertEqual([
      u'Boulevard de Lunel░░░░░░░░░░░░░░░░░░░',
      'Rue de Provence░░░░░░░░░░░░░░░░░░░░░░',
      'Rue de Port Maria░░░░░░░░░░░░░░░░░░░░',
      'Avenue Charles Gounod░░░░░░░░░░░░░░░░',
      'Rue de l‘Aurore░░░░░░░░░░░░░░░░░░░░░░',
      'Rue de Beuzeville░░░░░░░░░░░░░░░░░░░░',
      'Rue d‘Orbey░░░░░░░░░░░░░░░░░░░░░░░░░░',
      'Rue Victor Schoulcher░░░░░░░░░░░░░░░░',
      'Rue de la Gare░░░░░░░░░░░░░░░░░░░░░░░',
      'Rue des Tulipes░░░░░░░░░░░░░░░░░░░░░░',
      'Rue André Maginot░░░░░░░░░░░░░░░░░░░░',
      'Route de Pringy░░░░░░░░░░░░░░░░░░░░░░',
      'Rue des Landelles░░░░░░░░░░░░░░░░░░░░',
      'Rue des Ilettes░░░░░░░░░░░░░░░░░░░░░░',
      'Avenue de Maurin░░░░░░░░░░░░░░░░░░░░░',
      'Rue Théresa░░░░░░░░░░░░░░░░░░░░░░░░░░',  # GT='Rue Thérésa'
      'Route de la Balme░░░░░░░░░░░░░░░░░░░░',
      'Rue Hélène Roederer░░░░░░░░░░░░░░░░░░',
      'Rue Emile Bernard░░░░░░░░░░░░░░░░░░░░',
      'Place de la Mairie░░░░░░░░░░░░░░░░░░░',
      'Rue des Perrots░░░░░░░░░░░░░░░░░░░░░░',
      'Rue de la Libération░░░░░░░░░░░░░░░░░',
      'Impasse du Capcir░░░░░░░░░░░░░░░░░░░░',
      'Avenue de la Grand Mare░░░░░░░░░░░░░░',
      'Rue Pierre Brossolette░░░░░░░░░░░░░░░',
      'Rue de Provence░░░░░░░░░░░░░░░░░░░░░░',
      'Rue du Docteur Mourre░░░░░░░░░░░░░░░░',
      'Rue d‘Ortheuil░░░░░░░░░░░░░░░░░░░░░░░',
      'Rue des Sarments░░░░░░░░░░░░░░░░░░░░░',
      'Rue du Centre░░░░░░░░░░░░░░░░░░░░░░░░',
      'Impasse Pierre Mourgues░░░░░░░░░░░░░░',
      'Rue Marcel Dassault░░░░░░░░░░░░░░░░░░'
    ], predictions)


if __name__ == '__main__':
  tf.test.main()
