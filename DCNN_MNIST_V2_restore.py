import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
import numpy as np

#np.set_printoptions(threshold=np.inf)
import pickle
from tensorflow.python.platform import gfile
from tensorflow.examples.tutorials.mnist import input_data

model_filename_before_quantization = "saved_model/dcnn_mnist_v2.pb";
model_filename_after_quantization = "quantization_model/dcnn_mnist_v2_quantizate.pb";
model_filename = model_filename_after_quantization;

# with tf.Session() as sess:
#     with open(model_filename, 'rb') as f:
#         graph_def = tf.GraphDef()
#         graph_def.ParseFromString(f.read()) 
#         output = tf.import_graph_def(graph_def, return_elements=['W1_min:0'])
#         print(sess.run(output)) 

# with open(model_filename, 'rb') as f:
#     graph_def = tf.GraphDef()
#     graph_def.ParseFromString(f.read())
# output = tf.import_graph_def(graph_def)
# graph = tf.get_default_graph()
# for op in graph.get_operations():
# 	print(op.name, op.type)

#np.set_printoptions(threshold='nan')
mnist = input_data.read_data_sets('MNIST_data', one_hot=True, reshape=True)
X = tf.placeholder(tf.float32, [None, 784])
Y_ = tf.placeholder(tf.float32, [None, 10])
dropout = tf.placeholder(tf.float32)
with tf.Session() as sess:
    with open(model_filename, 'rb') as f: 
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read()) 
        X = tf.placeholder(tf.float32, [None, 784], name="X")
        output = tf.import_graph_def(graph_def, input_map={'X:0': X, 'Y_:0': Y_,'dropout:0': dropout}, 
        														return_elements=['Reshape_eightbit_quantize_X:0','W1_quint8_const:0',
        																			'first_conv_eightbit_quantized_conv:0','first_conv_eightbit_requantize:0',
        																			'first_conv:0','B1:0','Add:0','Y1_eightbit_quantize_Add:0',
        																			'Y1_eightbit_quantized:0','accuracy:0']) 
        batch_X, batch_Y = mnist.train.next_batch(1)
        # print(batch_X.shape)
        # print(batch_X.dtype)
		#print(type(batch_X))
		#np.savetxt('model_parameter/X:0', batch_X)
		#np.savetxt('model_parameter/Y:0', batch_Y)
        # print(sess.run(output, feed_dict={X: batch_X, #mnist.test.next_batch(64)[0],
        #                               Y_: batch_Y, #mnist.test.labels[0],
        #                               dropout: 1.}))
        output_list = sess.run(output, feed_dict={X: batch_X, #mnist.test.next_batch(64)[0],
                              Y_: batch_Y, #mnist.test.labels[0],
                              dropout: 1.})


print(output_list[1])
print('')
print(batch_X.shape)
print(batch_X.dtype)
# print(type(batch_X))
np.savetxt('model_parameter/X:0', batch_X, fmt='%-10.4f')
np.savetxt('model_parameter/Y:0', batch_Y, fmt='%-10.4f')

filename_list = ['Reshape_eightbit_quantize_X:0','W1_quint8_const:0',
				'first_conv_eightbit_quantized_conv:0','first_conv_eightbit_requantize:0',
				'first_conv:0','B1:0','Add:0','Y1_eightbit_quantize_Add:0',
				'Y1_eightbit_quantized:0','accuracy:0']
for op in zip(output_list,filename_list):
	print('')
	print(op[1])
	print(op[0].shape)
	data_type = op[0].dtype
	print(data_type)
	file_path = 'model_parameter/'+op[1]
	# np.savetxt(file_path, op[0])
	# print(type(op[0]))
	# np.savetxt('model_parameter/'+op[1], op[0])
	data = op[0].reshape((1,-1))
	np.savetxt(file_path, data, fmt='%-10.4f')
	# with open(file_path, 'wb') as output_file:
	# 	# I'm writing a header here just for the sake of readability
	#     # Any line starting with "#" will be ignored by numpy.loadtxt
	#     # output_file.write('# Array shape: {0}\n'.format(data.shape))

	#     # Iterating through a ndimensional array produces slices along
	#     # the last axis. This is equivalent to data[i,:,:] in this case
	#     for data_slice in data:

	#         # The formatting string indicates that I'm writing out
	#         # the values in left-justified columns 7 characters in width
	#         # with 2 decimal places.  
	#         np.savetxt(output_file, data_slice, fmt='%-10.4f')

	#         # Writing out a break to indicate different slices...
	#         # output_file.write('# New slice\n')
	



















# with tf.Session() as sess:
#     model_filename = "saved_model/dcnn_mnist_v2.pb" 
#     with gfile.FastGFile(model_filename, 'rb') as f:
#         graph_def = tf.GraphDef()
#         graph_def.ParseFromString(f.read())
#     result = tf.import_graph_def(graph_def, return_elements=["W1:0"])
#     print(sess.run(result))


# pb_file_path = "saved_model/dcnn_mnist_v2.pb"
# with tf.Graph().as_default():
#     output_graph_def = tf.GraphDef()

#     with open(pb_file_path, "rb") as f:
#         output_graph_def.ParseFromString(f.read())
#         _ = tf.import_graph_def(output_graph_def, name="")

# with tf.Session() as sess:
# 	init = tf.global_variables_initializer()
# 	sess.run(init)

# 	# input_x = sess.graph.get_tensor_by_name("input:0")
# 	# print input_x
# 	# out_softmax = sess.graph.get_tensor_by_name("softmax:0")
# 	# print out_softmax
# 	# out_label = sess.graph.get_tensor_by_name("output:0")
# 	# print out_label

# 	# graph = tf.get_default_graph()
# 	Y1 = sess.graph.get_operations()
# 	print(Y1)

# 	# Y_ = sess.graph.get_tensor_by_name("Y_:0")
# 	# dropout = sess.graph.get_tensor_by_name("dropout:0")
# 	# accuracy = sess.graph.get_tensor_by_name("accuracy:0")

# 	# mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
# 	# print("Testing Accuracy:", \
# 	# 	sess.run(accuracy, feed_dict={X: mnist.test.images,
# 	#                               		Y_: mnist.test.labels,
# 	#                               		dropout: 1.}))

# 	# img = io.imread(jpg_path)
# 	# img = transform.resize(img, (224, 224, 3))
# 	# img_out_softmax = sess.run(out_softmax, feed_dict={input_x:np.reshape(img, [-1, 224, 224, 3])})

# 	# print "img_out_softmax:",img_out_softmax
# 	# prediction_labels = np.argmax(img_out_softmax, axis=1)
# 	# print "label:",prediction_labels