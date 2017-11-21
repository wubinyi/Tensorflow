import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import math
from tensorflow.examples.tutorials.mnist import input_data as mnist_data
from tensorflow.python.framework import graph_util

X = tf.placeholder(tf.float32, [None, 25], name="X")
X = tf.reshape(X, shape=[-1, 5, 5, 1], name='reshape_X')

W1 = tf.Variable(tf.truncated_normal([3, 3, 1, 2], stddev=0.1), name="W1")
B1 = tf.Variable(tf.random_normal([2]), name="B1")


stride = 1  # output is 24x24
Y1l = tf.nn.conv2d(X, W1, strides=[1, stride, stride, 1], padding='VALID', name='first_conv')
Y1bias = tf.nn.bias_add(Y1l, B1, name='first_bias')
Y1 = tf.nn.relu(Y1bias, name="Y1")

init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	output_graph_def = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['Y1'])
	with tf.gfile.GFile("saved_model/cnn_sa_test.pb", "wb") as f:
		f.write(output_graph_def.SerializeToString())