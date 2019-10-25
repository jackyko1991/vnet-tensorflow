from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tqdm import tqdm

tf.reset_default_graph()
imported_meta = tf.train.import_meta_graph("./tmp/ckpt/checkpoint-154203.meta")

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
	print("Restoring checkpoint...")
	imported_meta.restore(sess, tf.train.latest_checkpoint("./tmp/ckpt",latest_filename="checkpoint-latest"))

	print("Restore checkpoint complete")

	print("Assign tensor to variables")
	for variable in tqdm(tf.trainable_variables()):
		tensor = tf.constant(variable.eval())
		tf.assign(variable, tensor, name="nWeights")

	# tf.train.write_graph(sess.graph.as_graph_def(), "D:/projects/Deep_Learning/tensorflow/vnet-tensorflow/tmp/", "graph_ascii.pb")

	print("Writing graph...")
	tf.train.write_graph(sess.graph.as_graph_def(), "./tmp/", "graph.pb",as_text=False)