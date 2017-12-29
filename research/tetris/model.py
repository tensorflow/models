import tensorflow as tf
import numpy as np

def weight_variable(shape, name=None):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial, name=name)

def bias_variable(shape, init=0.1, name=None):
	initial = tf.constant(init, shape=shape)
	return tf.Variable(initial, name=name)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x, name = None):
	return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name=name)

def max_pool_2x1(x, name = None):
	return tf.nn.max_pool(x, ksize=[1,2,1,1], strides=[1,2,1,1], padding='SAME', name=name)

def create_model_5():
	model = tf.Graph()
	with model.as_default():
		#input
		_from = tf.placeholder(tf.float32, [None, 20, 10], name="from") 
		_to = tf.placeholder(tf.float32, [None, 20, 10], name="to")
		_next = tf.placeholder(tf.int32, [None], name="next")
		keep_prob = tf.placeholder(tf.float32, name="kp")

		#from image
		W_conv_from = weight_variable([3,3,1,64])
		b_conv_from = bias_variable([64])
		h_conv_from = tf.nn.relu(conv2d(tf.reshape(_from, [-1, 20, 10, 1]), W_conv_from) + b_conv_from)
		h_pool_from = max_pool_2x1(h_conv_from)

		W_conv2_from = weight_variable([3,3,64,64])
		b_conv2_from = bias_variable([64])
		h_conv2_from = tf.nn.relu(conv2d(h_pool_from, W_conv2_from) + b_conv2_from)
		h_pool2_from = max_pool_2x2(h_conv2_from)

		h_from = tf.reshape(h_pool2_from, [-1, 5 * 5 * 64])

		#to image
		W_conv_to = weight_variable([3,3,1,64])
		b_conv_to = bias_variable([64])
		h_conv_to = tf.nn.relu(conv2d(tf.reshape(_to, [-1, 20, 10, 1]), W_conv_to) + b_conv_to)
		h_pool_to = max_pool_2x1(h_conv_to)

		W_conv2_to = weight_variable([3,3,64,64])
		b_conv2_to = bias_variable([64])
		h_conv2_to = tf.nn.relu(conv2d(h_pool_to, W_conv2_to) + b_conv2_to)
		h_pool2_to = max_pool_2x2(h_conv2_to)

		h_to = tf.reshape(h_pool2_to, [-1, 5 * 5 * 64])

		#next one hot
		onehot_next = tf.one_hot(_next, 7)

		#layer fc1
		W_fc1 = weight_variable([5 * 5 * 64 * 2 + 7, 1024])
		b_fc1 = bias_variable([1024])
		h_fc1_input = tf.concat([h_from, h_to, onehot_next], 1)
		h_fc1 = tf.nn.relu(tf.matmul(h_fc1_input, W_fc1) + b_fc1)
		print("W_fc1", W_fc1)
		print("h_fc1", h_fc1)

		#drop out
		h_drop = tf.nn.dropout(h_fc1, keep_prob)

		#layer out Q
		W_out_Q = weight_variable([1024, 1])
		b_out_Q = bias_variable([1])
		Q = tf.matmul(h_drop, W_out_Q) + b_out_Q
		output = tf.reshape(Q, [-1], name="output")
		print("output", output)

	return model

if __name__ == "__main__":
	create_model_5()
