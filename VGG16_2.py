from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# Parameters
learning_rate = 0.0005
training_iters = 20000
batch_size = 50
display_step = 100
filter_size = 3

# Network Parameters
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)
dropout = 0.5 # Dropout, probability to keep units

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)


# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (keep the size)
    conv2 = maxpool2d(conv2, k=1)

    # Convolution Layer
    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    # Convolution Layer
    conv4 = conv2d(conv3, weights['wc4'], biases['bc4'])
    # Max Pooling (down-sampling)
    conv4 = maxpool2d(conv4, k=2)

    # Convolution Layer
    conv5 = conv2d(conv4, weights['wc5'], biases['bc5'])
    # Convolution Layer
    conv6 = conv2d(conv5, weights['wc6'], biases['bc6'])
    # Convolution Layer
    conv7 = conv2d(conv6, weights['wc7'], biases['bc7'])
    # Max Pooling (down-sampling)
    conv7 = maxpool2d(conv7, k=2)

    # Fully connected layer
    # Reshape conv7 output to fit fully connected layer input
    fc1 = tf.reshape(conv7, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)

    # Fully connected layer
    fc2 = tf.add(tf.matmul(fc1, weights['wd2']), biases['bd2'])
    fc2 = tf.nn.relu(fc2)
    # Apply Dropout
    fc2 = tf.nn.dropout(fc2, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc2, weights['out']), biases['out'])
    return out

# Store layers weight & bias
weights = {
    # 'filter_size' conv, 1 input, 8 outputs
    # kernel size: filter_size * filter_size * 1
    'wc1': tf.Variable(tf.random_normal([filter_size, filter_size, 1, 8])),
    # 'filter_size' conv, 8 input, 8 outputs
    # kernel size: filter_size * filter_size * 8
    'wc2': tf.Variable(tf.random_normal([filter_size, filter_size, 8, 8])),

    # 'filter_size' conv, 8 input, 16 outputs
    # kernel size: filter_size * filter_size * 8
    'wc3': tf.Variable(tf.random_normal([filter_size, filter_size, 8, 16])),
    # 'filter_size' conv, 16 input, 16 outputs
    # kernel size: filter_size * filter_size * 16
    'wc4': tf.Variable(tf.random_normal([filter_size, filter_size, 16, 16])),

    # 'filter_size' conv, 16 input, 32 outputs
    # kernel size: filter_size * filter_size * 16
    'wc5': tf.Variable(tf.random_normal([filter_size, filter_size, 16, 32])),
    # 'filter_size' conv, 32 input, 32 outputs
    # kernel size: filter_size * filter_size * 32
    'wc6': tf.Variable(tf.random_normal([filter_size, filter_size, 32, 32])),
    # 'filter_size' conv, 32 input, 32 outputs
    # kernel size: filter_size * filter_size * 32
    'wc7': tf.Variable(tf.random_normal([filter_size, filter_size, 32, 32])),

    # 'filter_size' conv, 32 input, 64 outputs
    # kernel size: filter_size * filter_size * 32
    #'wc8': tf.Variable(tf.random_normal([filter_size, filter_size, 32, 64])),
    # 'filter_size' conv, 64 input, 64 outputs
    # kernel size: filter_size * filter_size * 64
    #'wc9': tf.Variable(tf.random_normal([filter_size, filter_size, 64, 64])),
    # 'filter_size' conv, 64 input, 64 outputs
    # kernel size: filter_size * filter_size * 64
    #'wc10': tf.Variable(tf.random_normal([filter_size, filter_size, 64, 64])),

    # 'filter_size' conv, 64 input, 64 outputs
    # kernel size: filter_size * filter_size * 32
    #'wc11': tf.Variable(tf.random_normal([filter_size, filter_size, 64, 64])),
    # 'filter_size' conv, 64 input, 64 outputs
    # kernel size: filter_size * filter_size * 64
    #'wc12': tf.Variable(tf.random_normal([filter_size, filter_size, 64, 64])),
    # 'filter_size' conv, 64 input, 64 outputs
    # kernel size: filter_size * filter_size * 64
    #'wc13': tf.Variable(tf.random_normal([filter_size, filter_size, 64, 64])),

    # fully connected, 7*7*32 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([7*7*32, 1024])),
    'wd2': tf.Variable(tf.random_normal([1024, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([8])),
    'bc2': tf.Variable(tf.random_normal([8])),
    'bc3': tf.Variable(tf.random_normal([16])),
    'bc4': tf.Variable(tf.random_normal([16])),
    'bc5': tf.Variable(tf.random_normal([32])),
    'bc6': tf.Variable(tf.random_normal([32])),
    'bc7': tf.Variable(tf.random_normal([32])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'bd2': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step < training_iters:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                       keep_prob: dropout})
        if step % display_step == 0:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={x: mnist.test.images[:50],# batch_x,
                                                              y: mnist.test.labels[:50],# batch_y,
                                                              keep_prob: 1.})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")

    # Calculate accuracy for 256 mnist test images
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: mnist.test.images[:256],
                                      y: mnist.test.labels[:256],
                                      keep_prob: 1.}))
    # save the model parameter
    saver=tf.train.Saver(max_to_keep=1)
    saver.save(sess,"D:\Tensorflow\Workspace\model.ckpt",global_step=0)
    # print variable "weights"
    counter = 1
    for weight in weights:
        print("=========================== weight in " + str(counter) + ". layer============================")
        counter = counter + 1
        print(sess.run(weights[weight]))
        print('\n')
    # print variable "biases"
    counter = 1
    for bias in biases:
        print("=========================== weight in " + str(counter) + ". layer============================")
        counter = counter + 1
        print(sess.run(biases[bias]))
        print('\n')