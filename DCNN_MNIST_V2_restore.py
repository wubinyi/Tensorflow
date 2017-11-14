import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
import numpy as np

#np.set_printoptions(threshold=np.inf)
import pickle
from tensorflow.python.platform import gfile
from tensorflow.examples.tutorials.mnist import input_data

model_filename_before_quantization = "saved_model/dcnn_mnist_v2.pb";
model_filename_after_quantization = "quantization_model/dcnn_mnist_v2_2_quantizate.pb";
model_filename = model_filename_after_quantization;

# print one 'variable', but now it is constant
# with tf.Session() as sess:
#     with open(model_filename, 'rb') as f:
#         graph_def = tf.GraphDef()
#         graph_def.ParseFromString(f.read()) 
#         output = tf.import_graph_def(graph_def, return_elements=['W1_min:0'])
#         print(sess.run(output)) 

# print operation name and operation type
# with open(model_filename, 'rb') as f:
#     graph_def = tf.GraphDef()
#     graph_def.ParseFromString(f.read())
# output = tf.import_graph_def(graph_def)
# graph = tf.get_default_graph()
# for op in graph.get_operations():
# 	print(op.name, op.type)

# use quantized mode to do inference work and get accuracy
# mnist = input_data.read_data_sets('MNIST_data', one_hot=True, reshape=True)
# X = tf.placeholder(tf.float32, [None, 784])
# Y_ = tf.placeholder(tf.float32, [None, 10])
# dropout = tf.placeholder(tf.float32)
# with tf.Session() as sess:
#     with open(model_filename, 'rb') as f: 
#         graph_def = tf.GraphDef()
#         graph_def.ParseFromString(f.read()) 
#         X = tf.placeholder(tf.float32, [None, 784], name="X")
#         output = tf.import_graph_def(graph_def, input_map={'X:0': X, 'Y_:0': Y_,'dropout:0': dropout}, 
# 														return_elements=['accuracy:0']) 
#         print(sess.run(output, feed_dict={X: mnist.test.images,
#                                       Y_: mnist.test.labels,
#                                       dropout: 1.}))

# use quantized mode or unquantized mode to do inference work
mnist = input_data.read_data_sets('MNIST_data', one_hot=True, reshape=True)
X = tf.placeholder(tf.float32, [None, 784])
Y_ = tf.placeholder(tf.float32, [None, 10])
dropout = tf.placeholder(tf.float32)
with tf.Session() as sess:
    with open(model_filename, 'rb') as f: 
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read()) 
        X = tf.placeholder(tf.float32, [None, 784], name="X")
        # version V2
        # output = tf.import_graph_def(graph_def, input_map={'X:0': X, 'Y_:0': Y_,'dropout:0': dropout}, 
        # 														return_elements=['Reshape_eightbit_quantize_X:0','W1_quint8_const:0',
        # 																			'first_conv_eightbit_quantized_conv:0','first_conv_eightbit_requantize:0',
        # 																			'first_conv:0','B1:0','Add:0','Y1_eightbit_quantize_Add:0',
        # 																			'Y1_eightbit_quantized:0','accuracy:0']) 
        # version V2_1
        # output = tf.import_graph_def(graph_def, input_map={'X:0': X, 'Y_:0': Y_,'dropout:0': dropout}, 
								# 						return_elements=['reshape_X_eightbit_quantize_X:0','W1_quint8_const:0',
								# 											'first_conv_eightbit_quantized_conv:0','first_conv_eightbit_requantize:0',
								# 											'B1_quint8_const:0','BiasAdd_eightbit_quantized_bias_add:0',
								# 											'BiasAdd_eightbit_requantize:0','Y1_eightbit_quantized:0','accuracy:0']) 
        # version V2_2
        output = tf.import_graph_def(graph_def, input_map={'X:0': X, 'Y_:0': Y_,'dropout:0': dropout}, 
														return_elements=['reshape_X_eightbit_quantize_X:0','W1_quint8_const:0',
																			'first_conv_eightbit_quantized_conv:0','first_conv_eightbit_requantize:0',
																			'B1_quint8_const:0','first_bias_eightbit_quantized_bias_add:0',
																			'first_bias_eightbit_requantize:0','Y1_eightbit_quantized:0','accuracy:0']) 
        batch_X, batch_Y = mnist.train.next_batch(1)
        # print(sess.run(output, feed_dict={X: batch_X, #mnist.test.next_batch(64)[0],
        #                               Y_: batch_Y, #mnist.test.labels[0],
        #                               dropout: 1.}))
        output_list = sess.run(output, feed_dict={X: batch_X, #mnist.test.next_batch(64)[0],
                              Y_: batch_Y,                    #mnist.test.labels[0],
                              dropout: 1.})
print('')
print(batch_X.shape)
print(batch_X.dtype)
# print(type(batch_X))
np.savetxt('model_parameter/V2_2/X:0', batch_X, fmt='%-10.4f')
np.savetxt('model_parameter/V2_2/Y:0', batch_Y, fmt='%-10.4f')
# version V2
# filename_list = ['Reshape_eightbit_quantize_X:0','W1_quint8_const:0',
# 				'first_conv_eightbit_quantized_conv:0','first_conv_eightbit_requantize:0',
# 				'first_conv:0','B1:0','Add:0','Y1_eightbit_quantize_Add:0',
# 				'Y1_eightbit_quantized:0','accuracy:0']
# version V2_1
# filename_list = ['reshape_X_eightbit_quantize_X:0','W1_quint8_const:0',
# 					'first_conv_eightbit_quantized_conv:0','first_conv_eightbit_requantize:0',
# 					'B1_quint8_const:0','BiasAdd_eightbit_quantized_bias_add:0',
# 					'BiasAdd_eightbit_requantize:0','Y1_eightbit_quantized:0','accuracy:0']
# version V2_2
filename_list = ['reshape_X_eightbit_quantize_X:0','W1_quint8_const:0',
					'first_conv_eightbit_quantized_conv:0','first_conv_eightbit_requantize:0',
					'B1_quint8_const:0','first_bias_eightbit_quantized_bias_add:0',
					'first_bias_eightbit_requantize:0','Y1_eightbit_quantized:0','accuracy:0']
for op in zip(output_list,filename_list):
	print('')
	print(op[1])
	print(op[0].shape)
	data_type = op[0].dtype
	print(data_type)
	file_path = 'model_parameter/V2_2/'+op[1]
	# print(type(op[0]))
	data = op[0].reshape((1,-1))
	np.savetxt(file_path, data, fmt='%-10.4f')