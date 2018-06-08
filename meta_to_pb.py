from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

tf.reset_default_graph()
imported_meta = tf.train.import_meta_graph("./tmp/ckpt/checkpoint-67591.meta")

with tf.Session() as sess:
	imported_meta.restore(sess, tf.train.latest_checkpoint("./tmp/ckpt",latest_filename="checkpoint-latest"))

	for variable in tf.trainable_variables():
		tensor = tf.constant(variable.eval())
		tf.assign(variable, tensor, name="nWeights")

	# tf.train.write_graph(sess.graph.as_graph_def(), "D:/projects/Deep_Learning/tensorflow/vnet-tensorflow/tmp/", "graph_ascii.pb")
	tf.train.write_graph(sess.graph.as_graph_def(), "D:/projects/Deep_Learning/tensorflow/vnet-tensorflow/tmp/", "graph.pb",as_text=False)