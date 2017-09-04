from __future__ import absolute_import, division, print_function

import os
import util
import ops
import models

import argparse
import numpy as np 
import tensorflow as tf 
import imgUtil

from glob import glob

''' Parse parameters '''
parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset', dest='dataset', default='horse2zebra', help='which dataset to use')
parser.add_argument('--crop_size', dest='crop_size', type=int, default=256, help='crop to size')
args = parser.parse_args()

dataset = args.dataset
crop_size = args.crop_size

''' run '''
with tf.Session() as sess:
	a_real = tf.placeholder(tf.float32, shape=[None, crop_size, crop_size, 3])
	b_real = tf.placeholder(tf.float32, shape=[None, crop_size, crop_size, 3])

	a2b = models.generator(a_real, 'a2b')
	b2a = models.generator(b_real, 'b2a')
	b2a2b = models.generator(b2a, 'a2b', reuse=True)
	a2b2a = models.generator(a2b, 'b2a', reuse=True)

	#restore
	saver = tf.train.Saver()
	ckpt_path = util.load_checkpoint('./checkpoints/' + dataset, sess, saver)
	if ckpt_path is None:
		raise Exception('No checkpoint!')
	else:
		print('Copy variables from %s' % ckpt_path)

	#test
	testA = glob('./datasets/' + dataset + '/testA/*.jpg')
	testB = glob('./datasets/' + dataset + '/testB/*.jpg')

	saveA_path = './test_predictions/' + dataset + '/testA/'
	saveB_path = './test_predictions/' + dataset + '/testB/'
	util.mkdir([saveA_path, saveB_path])

	for i in range(len(testA)):
		realA_ipt = imgUtil.imresize(imgUtil.imread(testA[i]), [crop_size, crop_size])
		realA_ipt.shape = 1, crop_size, crop_size, 3
		a2b_opt, a2b2a_opt = sess.run([a2b, a2b2a], feed_dict={a_real: realA_ipt})
		a_img_opt = np.concatenate((realA_ipt, a2b_opt, a2b2a_opt), axis=0)

		img_name = os.path.basename(testA[i])
		imgUtil.imsave(imgUtil.immerge(a_img_opt, 1, 3), saveA_path + img_name)
		print('Save %s' % (saveA_path + img_name))

	for i in range(len(testB)):
		realB_ipt = imgUtil.imresize(imgUtil.imread(testB[i]), [crop_size, crop_size])
		realB_ipt.shape = 1, crop_size, crop_size, 3
		b2a_opt, b2a2b_opt = sess.run([b2a, b2a2b], feed_dict={b_real: realB_ipt})
		b_img_opt = np.concatenate((realB_ipt, b2a_opt, b2a2b_opt), axis=0)

		img_name = os.path.basename(testB[i])
		imgUtil.imsave(imgUtil.immerge(b_img_opt, 1, 3), saveB_path + img_name)
		print('Save %s' % (saveB_path + img_name))