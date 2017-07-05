import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import math
from tensorflow.examples.tutorials.mnist import input_data as mnist_data

# Download images and labels into mnist.test (10K images+labels) and mnist.train (60K images+labels)
mnist = mnist_data.read_data_sets("data", one_hot=True, reshape=False, validation_size=200)

# neural network structure for this sample:
#
# · · · · · · · · · ·      (input data, 1-deep)                    X [batch, 28, 28, 1]
# @ @ @ @ @ @ @ @ @ @   -- conv. layer +BN 6x6x1=>24 stride 1      W1 [5, 5, 1, 24]        B1 [24]
# ∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶                                              Y1 [batch, 28, 28, 6]
#   @ @ @ @ @ @ @ @     -- conv. layer +BN 5x5x6=>48 stride 2      W2 [5, 5, 6, 48]        B2 [48]
#   ∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶                                                Y2 [batch, 14, 14, 12]
#     @ @ @ @ @ @       -- conv. layer +BN 4x4x12=>64 stride 2     W3 [4, 4, 12, 64]       B3 [64]
#     ∶∶∶∶∶∶∶∶∶∶∶                                                  Y3 [batch, 7, 7, 24] => reshaped to YY [batch, 7*7*24]
#      \x/x\x\x/ ✞      -- fully connected layer (relu+dropout+BN) W4 [7*7*24, 200]       B4 [200]
#       · · · ·                                                    Y4 [batch, 200]
#       \x/x\x/         -- fully connected layer (softmax)         W5 [200, 10]           B5 [10]
#        · · ·                                                     Y [batch, 10]

# input X: 28x28 grayscale images, the first dimension (None) will index the images in the mini-batch
X = tf.placeholder(tf.float32, [None, 28, 28, 1])
# correct answers will go here
Y_ = tf.placeholder(tf.float32, [None, 10])
# variable learning rate
lr = tf.placeholder(tf.float32)
# test flag for batch norm
tst = tf.placeholder(tf.bool)
iter = tf.placeholder(tf.int32)
# dropout probability
dropout = tf.placeholder(tf.float32)

def batchnorm(Ylogits, is_test, iteration, offset, convolutional=False):
    exp_moving_avg = tf.train.ExponentialMovingAverage(0.999, iteration) # adding the iteration prevents from averaging across non-existing iterations
    bnepsilon = 1e-5
    if convolutional:
        mean, variance = tf.nn.moments(Ylogits, [0, 1, 2])
    else:
        mean, variance = tf.nn.moments(Ylogits, [0])
    update_moving_everages = exp_moving_avg.apply([mean, variance])
    m = tf.cond(is_test, lambda: exp_moving_avg.average(mean), lambda: mean)
    v = tf.cond(is_test, lambda: exp_moving_avg.average(variance), lambda: variance)
    Ybn = tf.nn.batch_normalization(Ylogits, m, v, offset, None, bnepsilon)
    return Ybn, update_moving_everages


# three convolutional layers with their channel counts, and a
# fully connected layer (tha last layer has 10 softmax neurons)
K = 24  # first convolutional layer output depth
L = 48  # second convolutional layer output depth
M = 64  # third convolutional layer
N = 200  # fully connected layer

W1 = tf.Variable(tf.random_normal([6, 6, 1, K]))  # 6x6 patch, 1 input channel, K output channels
W2 = tf.Variable(tf.random_normal([5, 5, K, L]))
W3 = tf.Variable(tf.random_normal([4, 4, L, M]))
W4 = tf.Variable(tf.random_normal([7 * 7 * M, N]))
W5 = tf.Variable(tf.random_normal([N, 10]))

B1 = tf.Variable(tf.random_normal([K]))
B2 = tf.Variable(tf.random_normal([L]))
B3 = tf.Variable(tf.random_normal([M]))
B4 = tf.Variable(tf.random_normal([N]))
B5 = tf.Variable(tf.random_normal([10]))


# The model
# batch norm scaling is not useful with relus
# batch norm offsets are used instead of biases
stride = 1  # output is 28x28
Y1l = tf.nn.conv2d(X, W1, strides=[1, stride, stride, 1], padding='SAME')
Y1bn, update_ema1 = batchnorm(Y1l, tst, iter, B1, convolutional=True)
Y1 = tf.nn.relu(Y1bn)

stride = 2  # output is 14x14
Y2l = tf.nn.conv2d(Y1, W2, strides=[1, stride, stride, 1], padding='SAME')
Y2bn, update_ema2 = batchnorm(Y2l, tst, iter, B2, convolutional=True)
Y2 = tf.nn.relu(Y2bn)

stride = 2  # output is 7x7
Y3l = tf.nn.conv2d(Y2, W3, strides=[1, stride, stride, 1], padding='SAME')
Y3bn, update_ema3 = batchnorm(Y3l, tst, iter, B3, convolutional=True)
Y3 = tf.nn.relu(Y3bn)


# reshape the output from the third convolution for the fully connected layer
Y3 = tf.reshape(Y3, shape=[-1, 7 * 7 * M])

Y4l = tf.matmul(Y3, W4)
Y4bn, update_ema4 = batchnorm(Y4l, tst, iter, B4)
Y4r = tf.nn.relu(Y4bn)
Y4 = tf.nn.dropout(Y4r, dropout)

Ylogits = tf.matmul(Y4, W5) + B5
#Y = tf.nn.softmax(Ylogits)

update_ema = tf.group(update_ema1, update_ema2, update_ema3, update_ema4)




# cross-entropy loss function (= -sum(Y_i * log(Yi)) ), normalised for batches of 100  images
# TensorFlow provides the softmax_cross_entropy_with_logits function to avoid numerical stability
# problems with log(0) which is NaN
#cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
#cross_entropy = tf.reduce_mean(cross_entropy)*100
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_))*100

# accuracy of the trained model, between 0 (worst) and 1 (best)
#Y = tf.nn.softmax(Ylogits)
correct_prediction = tf.equal(tf.argmax(Ylogits, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# training step, the learning rate is a placeholder
optimizer = tf.train.AdamOptimizer(lr).minimize(cost)

# init
init = tf.global_variables_initializer()



def training_step(i):

    # training on batches of 100 images with 100 labels
    batch_X, batch_Y = mnist.train.next_batch(100)

    # learning rate decay
    max_learning_rate = 0.02
    min_learning_rate = 0.0001
    decay_speed = 1600
    learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-i/decay_speed)
    #learning_rate = 0.0001

    # the backpropagation training step
    sess.run(optimizer, feed_dict={X: batch_X, Y_: batch_Y, lr: learning_rate, tst: False, dropout: 0.75})
    sess.run(update_ema, feed_dict={X: batch_X, Y_: batch_Y, tst: False, iter: i, dropout: 1.0})


# Launch the graph
config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step < 6001:

        training_step(step)

        if step % 100 == 0:
            # Calculate batch loss and accuracy
            vali_batch_x = mnist.validation.images
            vali_batch_y = mnist.validation.labels
            loss, acc = sess.run([cost, accuracy], feed_dict={X: vali_batch_x, Y_: vali_batch_y, tst: True, dropout: 1.0})
            print("Iter " + str(step) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Validation Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1


    print("Optimization Finished!")
    # Calculate accuracy for 256 mnist test images
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: mnist.test.images,
                                      Y_: mnist.test.labels,
                                      tst: True,
                                      dropout: 1.}))
    # save the model parameter
    saver=tf.train.Saver(max_to_keep=1)
    saver.save(sess,"DCNN_MNIST_model.ckpt",global_step=0)