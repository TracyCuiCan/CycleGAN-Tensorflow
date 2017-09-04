from __future__ import division, print_function, absolute_import

import os
import copy
import numpy as np
import tensorflow as tf

def mkdir(paths):
	if not isinstance(paths, list):
		paths = [paths]
	for path in paths:
		path_dir, _ = os.path.split(path)
		if not os.path.isdir(path_dir):
			os.makedirs(path_dir)

def load_checkpoint(checkpoint_dir, sess, saver):
	print(" [*] Loading checkpoint...")
	ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
	if ckpt and ckpt.model_checkpoint_path:
		ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
		ckpt_path = os.path.join(checkpoint_dir, ckpt_name)
		saver.restore(sess, ckpt_path)
		print(" [*] Load SUCCESS")
		return ckpt_path
	else:
		print(" [*] Load FAILURE")
		return None

def counter(scope='counter'):
	with tf.variable_scope(scope):
		counter = tf.Variable(0, dtype=tf.int32, name='counter')
		update_counter = tf.assign(counter, tf.add(counter, 1))
		return counter, update_counter
