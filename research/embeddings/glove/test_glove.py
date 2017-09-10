import tensorflow as tf
import threading

glove = tf.load_op_library('./glove_ops.so')


class GloveTest(tf.test.TestCase):

    def testCoocurrenceMatrix(self):
        filename = 'test_data/testfile'
        window_size = 5
        min_count = 0
        batch_size = 6

        with self.test_session():
            (vocab_word, indices, values, words_per_epoch,
             current_epoch, num_processed_words, inputs,
             labels, ccounts) = glove.glove_model(filename,
                                                  batch_size,
                                                  window_size,
                                                  min_count)
            vocab_size = tf.shape(vocab_word)[0]
            indices_size = tf.shape(indices)[0]
            values_size = tf.shape(values)[0]


            self.assertEqual(vocab_size.eval(), 6)
            self.assertEqual(indices_size.eval(), 21)
            self.assertEqual(values_size.eval(), 21)
            self.assertEqual(words_per_epoch.eval(), len(indices.eval()))

            I = word2id[b'I']
            like = word2id[b'like']
            machine = word2id[b'machine']
            learning = word2id[b'learning']
            programming = word2id[b'programming']

            expected_values = {
                (I, like): 1.0, (I, machine): 0.5, (I, learning): 0.33333334,
                (I, programming): 0.45, (like, I): 1.0, (like, machine): 1.0,
                (like, learning): 0.5, (like, programming): 0.58333337,
                (machine, I): 0.5, (machine, like): 1.0,
                (machine, learning): 1.0, (machine, programming): 0.83333331,
                (learning, I): 0.33333334, (learning, like): 0.5,
                (learning, machine): 1.0, (learning, programming): 1.5,
                (programming, I): 0.45, (programming, like): 0.58333337,
                (programming, machine): 0.83333331,
                (programming, learning): 1.5, (programming, programming): 2.0
            }

            self.assertEqual(len(expected_values), len(indices.eval()))
            for index, value in zip(indices.eval(), values.eval()):
                self.assertAlmostEqual(
                    expected_values[tuple(index)], value)

    def testBatchExamples(self):
        filename = 'test_data/testfile'
        window_size = 5
        min_count = 0
        batch_size = 5
        concurrent_steps = 1

        (vocab_word, indices, values, words_per_epoch,
         current_epoch, num_processed_words, inputs,
         labels, ccounts) = glove.glove_model(filename,
                                              batch_size,
                                              window_size,
                                              min_count)

        sess = tf.Session()
        t_indices = sess.run(indices).tolist()
        t_values = sess.run(values).tolist()
        expected_epoch = 1

        def test_body():
            inputs_, labels_, ccounts_, epoch = sess.run(
                [inputs, labels, ccounts, current_epoch])

            for word, label, ccount in zip(inputs_, labels_, ccounts_):
                pos = t_indices.index([word, label])
                self.assertEqual(t_values[pos], ccount)

        workers = []

        for _ in range(concurrent_steps):
            t = threading.Thread(target=test_body)
            t.start()
            workers.append(t)

        for t in workers:
            t.join()

        curr_epoch = sess.run(current_epoch)
        self.assertEqual(expected_epoch, curr_epoch)


if __name__ == '__main__':
    tf.test.main()
