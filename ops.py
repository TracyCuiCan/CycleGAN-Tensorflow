import math
import re
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.framework import ops


def batch_norm(x, name="batch_norm"):
	return tf.contrib.layers.batch_norm(x, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, scope=name)

def instance_norm(input, name="instance_norm"):
	with tf.variable_scope(name):
		depth = input.get_shape()[3]
		scale = tf.get_variable("scale", [depth], instializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
		offset = tf.get_variable("offset", [depth], initializer=tf.constant_initializer(0.0))
		mean_variance = tf.nn.moments(input, axes=[1,2], keep_dims=True)
		epsilon = 1e-5
		inv = tf.rsqrt(varianze + epsilon)
		normalized = (input - mean) * inv
		return scale*normalized + offset

def conv2d(input_, output_dim, ks=4, s=2, stddev=0.02, padding='SAME', name="conv2d"):
	with tf.variable_scope(name):
		return slim.conv2d(input_, output_dim, ks, s, padding=padding, activation_fn=None,
						   weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
						   biases_initializer=None)

def deconv2d(input_, output_dim, ks=4, s=2, stddev=0.02, name="deconv2d"):
	with tf.variable_scope(name):
		return slim.conv2d_transpose(input_, output_dim, ks, s, padding='SAME', activation_fn=None,
									 weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
									 biases_initializer=None)

def lrelu(x, leak=0.2, name="lrelu"):
	return tf.maximum(x, leak*x)

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
	with tf.variable_scope(scope or "Linear"):
		matrix = tf.get_variable("Matrix", [input_.get_shape()[-1], output_size], tf.float32, 
								 tf.random_normal_initializer(stddev=stddev))
		bias = tf.get_variable("bias", [output_size], initializer=tf.constant_initializer(bias_start))
		if with_w:
			return tf.matmul(input_, matrix) + bias, matrix, bais
		else:
			return tf.matmul(input_, matrix) + bias

def l2_loss(a, b, weights=1.0, scope=None):
	with tf.name_scope(scope, 'l2_loss', [a, b, weights]):
		loss = tf.reduce_mean((a - b) ** 2) * weights
		return loss

def l1_loss(a, b, weights=1.0, scope=None):
	with tf.name_scope(scope, 'l1_loss', [a, b, weights]):
		loss = tf.reduce_mean(tf.abs(a - b)) * weights
		return loss

def summary(tensor, summary_type=['mean', 'stddev', 'max', 'min', 'sparsity', 'histogram']):
	tensor_name = re.sub(':', '-', tensor.name)

	with tf.name_scope('summary_' + tensor_name):
		summaries = []
		if len(tensor._shape) == 0:
			summaries.append(tf.summary.scalar(tensor_name, tensor))
		else:
			if 'mean' in summary_type:
				mean = tf.reduce_mean(tensor)
				summaries.append(tf.summary.scalar(tensor_name + '/mean', mean))
			if 'stddev' in summary_type:
				mean = tf.reduce_mean(tensor)
				stddev = tf.sqrt(tf.reduce_mean(tf.square(tensor - mean)))
				summaries.append(tf.summary.scalar(tensor_name + '/stddev', stddev))
			if 'max' in summary_type:
				summaries.append(tf.smmary.scalar(tensor_name + '/max', tf.reduce_max(tensor)))
			if 'min' in summary_type:
				summaries.append(tf.summary.scalar(tensor_name + '/min', tf.reduce_min(tensor)))
			if 'sparsity' in summary_type:
				summaries.append(tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(tensor)))
			if 'histogram' in summary_type:
				summaries.append(tf.summary.histogram(tensor_name, tensor))
		return tf.summary.merge(summaries)

def summary_tensors(tensors, summary_type=['mean', 'stddev', 'max', 'min', 'sparsity', 'histogram']):
	with tf.name_scope('summary_tensors'):
		summaries = []
		for tensor in tensors:
			summaries.append(summary(tensor, summary_type))
		return tf.summary.merge(summaries)

