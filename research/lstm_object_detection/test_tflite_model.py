from absl import flags
import numpy as np
import tensorflow as tf

flags.DEFINE_string('model_path', None, 'Path to model.')
FLAGS = flags.FLAGS


def main(_):

    flags.mark_flag_as_required('model_path')

    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=FLAGS.model_path)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    print 'input_details:', input_details
    output_details = interpreter.get_output_details()
    print 'output_details:', output_details

    # Test model on random input data.
    input_shape = input_details[0]['shape']
    # change the following line to feed into your own data.
    input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print output_data


if __name__ == '__main__':
    tf.app.run()
