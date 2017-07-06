import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf

# bias = tf.Variable(tf.random_normal([16]))
# init = tf.global_variables_initializer()

from tensorflow.examples.tutorials.mnist import input_data


# with tf.Session() as sess:
# 	sess.run(init)
# 	print(bias.get_shape().as_list()[0])
# 	vali_batch_x, vali_batch_y = mnist.validation.next_batch(1)
# 	print(vali_batch_y[0])
with tf.Session() as sess:
    #sess.run(init)
    saver = tf.train.import_meta_graph('DCNN_MNIST_BN_SMALLER_FILTER_model.meta')
    #saver.restore(sess,tf.train.latest_checkpoint('./'))
    saver.restore(sess,'DCNN_MNIST_BN_SMALLER_FILTER_model')

    graph = tf.get_default_graph()
    X = graph.get_tensor_by_name("X:0")
    Y_ = graph.get_tensor_by_name("Y_:0")
    tst = graph.get_tensor_by_name("tst:0")
    dropout = graph.get_tensor_by_name("dropout:0")
    accuracy = graph.get_tensor_by_name("accuracy:0")
    #W1 = graph.get_tensor_by_name("W1:0")
    #print(sess.run(W1))

    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: mnist.test.images,
                                      Y_: mnist.test.labels,
                                      tst: True,
                                      dropout: 1.}))