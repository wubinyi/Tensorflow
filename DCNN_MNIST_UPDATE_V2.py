import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import math
from tensorflow.examples.tutorials.mnist import input_data as mnist_data
from tensorflow.python.framework import graph_util

# Download images and labels into mnist.test (10K images+labels) and mnist.train (60K images+labels)
mnist = mnist_data.read_data_sets("data", one_hot=True, reshape=False, validation_size=200)
#mnist = mnist_data.read_data_sets("data", one_hot=True, validation_size=200)

# neural network structure for this sample:
#
# · · · · · · · · · ·      (input data, 1-deep)                    X [batch, 28, 28, 1]
# @ @ @ @ @ @ @ @ @ @   -- conv. layer 6x6x1=>24 stride 1      W1 [5, 5, 1, 24]        B1 [24]
# ∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶                                              Y1 [batch, 28, 28, 6]
#   @ @ @ @ @ @ @ @     -- conv. layer 5x5x6=>48 stride 2      W2 [5, 5, 6, 48]        B2 [48]
#   ∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶                                                Y2 [batch, 14, 14, 12]
#     @ @ @ @ @ @       -- conv. layer 4x4x12=>64 stride 2     W3 [4, 4, 12, 64]       B3 [64]
#     ∶∶∶∶∶∶∶∶∶∶∶                                                  Y3 [batch, 7, 7, 24] => reshaped to YY [batch, 7*7*24]
#      \x/x\x\x/ ✞      -- fully connected layer (relu+dropout+BN) W4 [7*7*24, 200]       B4 [200]
#       · · · ·                                                    Y4 [batch, 200]
#       \x/x\x/         -- fully connected layer (softmax)         W5 [200, 10]           B5 [10]
#        · · ·                                                     Y [batch, 10]

# input X: 28x28 grayscale images, the first dimension (None) will index the images in the mini-batch
#X = tf.placeholder(tf.float32, [None, 28, 28, 1], name="X")
X = tf.placeholder(tf.float32, [None, 784], name="X")
# correct answers will go here
Y_ = tf.placeholder(tf.float32, [None, 10], name="Y_")
# variable learning rate
lr = tf.placeholder(tf.float32, name="lr")
# dropout probability
dropout = tf.placeholder(tf.float32, name="dropout")


# three convolutional layers with their channel counts, and a
# fully connected layer (tha last layer has 10 softmax neurons)
#J = 16
K = 6  # first convolutional layer output depth
L = 12  # second convolutional layer output depth
M = 24  # third convolutional layer
N = 200  # fully connected layer

# kernel size: 6->5->4    accuracy: 
# kernel size: 5->4->3    accuracy:   99.15
# kernel size: 3->4->5    accuracy: 
# kernel size: 4->4->5    accuracy: 
# kernel size: 4->4->3    accuracy: 
# kernel size: 5->4->3    accuracy:  (add dropconnect)
# kernel size: 3->5->4->3 accuracy: 

# W1 = tf.Variable(tf.random_normal([5, 5, 1, K]), name="W1")  # 6x6 patch, 1 input channel, K output channels
# W2 = tf.Variable(tf.random_normal([4, 4, K, L]), name="W2")
# W3 = tf.Variable(tf.random_normal([3, 3, L, M]), name="W3")
# W4 = tf.Variable(tf.random_normal([7 * 7 * M, N]), name="W4")
# W5 = tf.Variable(tf.random_normal([N, 10]), name="W5")

W1 = tf.Variable(tf.truncated_normal([5, 5, 1, K], stddev=0.1), name="W1")
W2 = tf.Variable(tf.truncated_normal([4, 4, K, L], stddev=0.1), name="W2")
W3 = tf.Variable(tf.truncated_normal([3, 3, L, M], stddev=0.1), name="W3")
W4 = tf.Variable(tf.truncated_normal([19 * 19 * M, N], stddev=0.1), name="W4")
W5 = tf.Variable(tf.truncated_normal([N, 10], stddev=0.1), name="W5")

# B1 = tf.Variable(tf.random_normal([K]), name="B1")
# B2 = tf.Variable(tf.random_normal([L]), name="B2")
# B3 = tf.Variable(tf.random_normal([M]), name="B3")
# B4 = tf.Variable(tf.random_normal([N]), name="B4")
# B5 = tf.Variable(tf.random_normal([10]), name="B5")

B1 = tf.Variable(tf.constant(0.1, tf.float32, [K]), name="B1")
B2 = tf.Variable(tf.constant(0.1, tf.float32, [L]), name="B2")
B3 = tf.Variable(tf.constant(0.1, tf.float32, [M]), name="B3")
B4 = tf.Variable(tf.constant(0.1, tf.float32, [N]), name="B4")
B5 = tf.Variable(tf.constant(0.1, tf.float32, [10]), name="B5")

# The model
X = tf.reshape(X, shape=[-1, 28, 28, 1], name='reshape_X')

stride = 1  # output is 24x24
Y1l = tf.nn.conv2d(X, W1, strides=[1, stride, stride, 1], padding='VALID', name='first_conv')
Y1bias = tf.nn.bias_add(Y1l, B1, name='first_bias')
Y1 = tf.nn.relu(Y1bias, name="Y1")

stride = 1  # output is 21x21
Y2l = tf.nn.conv2d(Y1, W2, strides=[1, stride, stride, 1], padding='VALID', name='second_conv')
Y2bias = tf.nn.bias_add(Y2l, B2, name='second_bias')
Y2 = tf.nn.relu(Y2bias, name="Y2")

stride = 1  # output is 19x19
Y3l = tf.nn.conv2d(Y2, W3, strides=[1, stride, stride, 1], padding='VALID', name='third_onv')
Y3bias = tf.nn.bias_add(Y3l, B3, name='third_bias')
Y3 = tf.nn.relu(Y3bias, name="Y3")

# reshape the output from the third convolution for the fully connected layer
Y3 = tf.reshape(Y3, shape=[-1, 19 * 19 * M], name='reshape_after_conv')

Y4l = tf.matmul(Y3, W4, name='full_connect_multi')
Y4r = tf.nn.relu(tf.add(Y4l, B4), name='fully_connect')
Y4 = tf.nn.dropout(Y4r, dropout, name="Y4")

Ylogits = tf.add(tf.matmul(Y4, W5), B5, name="Y5")
#Y = tf.nn.softmax(Ylogits)


# cross-entropy loss function (= -sum(Y_i * log(Yi)) ), normalised for batches of 100  images
# TensorFlow provides the softmax_cross_entropy_with_logits function to avoid numerical stability
# problems with log(0) which is NaN
#cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
#cross_entropy = tf.reduce_mean(cross_entropy)*100
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_))
cost = tf.multiply(cost, 64, name="cost")

# accuracy of the trained model, between 0 (worst) and 1 (best)
#Y = tf.nn.softmax(Ylogits)
correct_prediction = tf.equal(tf.argmax(Ylogits, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")

# training step, the learning rate is a placeholder
optimizer = tf.train.AdamOptimizer(lr).minimize(cost,name="optimizer")

# init
init = tf.global_variables_initializer()



def training_step(i):

    # training on batches of 100 images with 100 labels
    batch_X, batch_Y = mnist.train.next_batch(64)

    # learning rate decay
    max_learning_rate = 0.003
    min_learning_rate = 0.0001
    decay_speed = 2000.0
    learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-i/decay_speed)
    #learning_rate = 0.0001

    # the backpropagation training step
    sess.run(optimizer, feed_dict={X: batch_X, Y_: batch_Y, lr: learning_rate, dropout: 0.75})





# Launch the graph
#config = tf.ConfigProto()
#config.gpu_options.allocator_type = 'BFC'
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step < 14001: #401:

        training_step(step)

        if step % 100 == 0:
            # Calculate batch loss and accuracy
            vali_batch_x = mnist.validation.images
            vali_batch_y = mnist.validation.labels
            loss, acc = sess.run([cost, accuracy], feed_dict={X: vali_batch_x, Y_: vali_batch_y, dropout: 1.0})
            print("Iter " + str(step) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Validation Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1


    print("Optimization Finished!")
    # save the model parameter 
    # saver=tf.train.Saver()
    # saver.save(sess,"DCNN_MNIST_UPDATE_V2_model")
    # as model as .pb
    # graph_def = tf.get_default_graph().as_graph_def()
    output_graph_def = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['accuracy'])
    with tf.gfile.GFile("saved_model/dcnn_mnist_v2_2.pb", "wb") as f:
           f.write(output_graph_def.SerializeToString())

    # Calculate accuracy for mnist test images
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: mnist.test.images,
                                      Y_: mnist.test.labels,
                                      dropout: 1.}))