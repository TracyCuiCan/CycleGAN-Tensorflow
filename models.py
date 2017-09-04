from __future__ import division
import tensorflow as tf 
from ops import *
from imgUtil import *

def discriminator(image, scope, df_dim=64, reuse=False):
	with tf.variable_scope(scope + '_discriminator'):
		#image is 256 x 256 x input_c_dim
		if reuse:
			tf.get_variable_scope().reuse_variables()
		else:
			assert tf.get_variable_scope().reuse is False

		h0 = lrelu(conv2d(image, df_dim, name='d_h0_conv')) # h0 is (128 x 128 x df_dim)
		h1 = lrelu(instance_norm(conv2d(h0, df_dim*2, name='d_h1_conv'), 'd_bn1')) # h1 is (64 x 64 x df_dim*2)
		h2 = lrelu(instance_norm(conv2d(h1, df_dim*4, name='d_h2_conv'), 'd_bn2')) # h2 is (32 x 32 x df_dim*4)
		h3 = lrelu(instance_norm(conv2d(h2, df_dim*8, s = 1, name='d_h3_conv'), 'd_bn3')) # h3 is (32 x 32 x df_dim*8)
		h4 = conv2d(h3, 1, s=1, name='d_h3_pred') #h4 is (32 x 32 x 1)
		return h4


def generator(image, scope, gf_dim=64, reuse=False):
	with tf.variable_scope(scope + '_generator'):
		if reuse:
			tf.get_variable_scope().reuse_variables();
		else:
			assert tf.get_variable_scope().reuse is False

		def residule_block(x, dim, name="res"):
			y = tf.pad(x, [[0,0], [1,1], [1,1], [0,0]], "REFLECT")
			y = instance_norm(conv2d(y, dim, 3, 1, padding='VALID', name=name+'_c1'), name+'_bn1')
			y = tf.pad(tf.nn.relu(y), [[0,0], [1,1], [1,1], [0,0]], "REFLECT")
			y = instance_norm(conv2d(y, dim, 3, 1, padding='VALID', name=name+'_c2'), name+'_bn2')
			return y + x

		c0 = tf.pad(image, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
		c1 = tf.nn.relu(instance_norm(conv2d(c0, gf_dim, 7, 1, padding='VALID', name='g_e1_c'), 'g_e1_bn'))
		c2 = tf.nn.relu(instance_norm(conv2d(c1, gf_dim*2, 3, 2, name='g_e2_c'), 'g_e2_bn'))
		c3 = tf.nn.relu(instance_norm(conv2d(c2, gf_dim*4, 3, 2, name='g_e3_c'), 'g_e3_bn'))

		r1 = residule_block(c3, gf_dim*4, name='g_r1')
		r2 = residule_block(r1, gf_dim*4, name='g_r2')
		r3 = residule_block(r2, gf_dim*4, name='g_r3')
		r4 = residule_block(r3, gf_dim*4, name='g_r4')
		r5 = residule_block(r4, gf_dim*4, name='g_r5')
		r6 = residule_block(r5, gf_dim*4, name='g_r6')
		r7 = residule_block(r6, gf_dim*4, name='g_r7')
		r8 = residule_block(r7, gf_dim*4, name='g_r8')
		r9 = residule_block(r8, gf_dim*4, name='g_r9')

		d1 = tf.nn.relu(instance_norm(deconv2d(r9, gf_dim*2, 3, 2, name='g_d1_dc'), name='g_d1_bn'))
		d2 = tf.nn.relu(instance_norm(deconv2d(d1, gf_dim, 3, 2, name='g_d2_dc'), name='g_d2_bn'))
		d2 = tf.pad(d2, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
		pred = tf.nn.tanh(conv2d(d2, 3, 7, 1, padding='VALID', name='g_pred_c'))

		return pred




