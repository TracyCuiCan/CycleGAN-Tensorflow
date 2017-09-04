import tensorflow as tf
import numpy as np
import scipy.misc

def transform(image, min_value=0., max_value=1., dtype=None):
	assert \
		np.min(image) >= -1.0 - 1e-5 and np.max(image) <= 1.0 + 1e-5 \
		and (image.dtype == np.float32 or image.dtype == np.float64), \
		'Input image should be float64 or float32 in range [-1., 1.]!'
	if dtype is None: dtype = image.dtype
	return ((image + 1.) / 2. * (max_value - min_value) + min_value).astype(dtype)

def im2int(image):
	'''transform images from float [-1., 1.] to uint8 [0, 255]'''
	return transform(image, 0, 255, np.uint8)

def im2float(image):
	'''transform images from int [0, 255] to float [-1., 1.]'''
	return transform(image, 0.0, 1.0)

def imread(path, is_grayscale = False):
	'''read an image into [-1., 1.] of float64'''
	if (is_grayscale):
		return scipy.misc.imread(path, flatten=True) / 127.5 - 1
	else:
		return scipy.misc.imread(path, mode='RGB') / 127.5 - 1

def imsave(image, path):
	'''save an [-1., 1.] image'''
	return scipy.misc.imsave(path, transform(image, 0, 255, np.uint8))

def imresize(image, size, interp='bilinear'):
	'''
	Resize an [-1., 1.] image

	size: int, float or tuple
		* int   - percentage of current size
		* float - fraction of current size
		* tuple - output size

	interp: str, optional
		Interpolation for resizing ('nearest', 'lanczos', 'bilinear', 'bicubic', 'cubic')
	'''
	return (scipy.misc.imresize(im2int(image), size, interp=interp) / 127.5 - 1).astype(image.dtype)

def immerge(images, row, col):
	if images.ndim == 4:
		c = images.shape[3]
	elif images.ndim == 3:
		c = 1

	h, w = images.shape[1], images.shape[2]
	if c > 1:
		img = np.zeros((h * row, w * col, c))
	else:
		img = np.zeros((h * row, w * col))
	for idx, image in enumerate(images):
		i = idx % col
		j = idx // col
		img[j*h:j*h+h, i*w:i*w+w, :] = image
	return img

def read_images(path_list, is_grayscale=False):
	images = [imread(path, is_grayscale) for path in path_list]
	return np.array(images)

def center_crop(x, crop_h, crop_w, resize_h=64, resize_w=64):
	if crop_w is None:
		crop_w = crop_h
	h, w = x.shape[:2]
	j = int(round((h - crop_h) / 2.))
	i = int(round((w - crop_w) / 2.))
	return scipy.misc.imresize(
		x[j:j+crop_h, i:i+crop_2], [resize_h, resize_w])



class ImagePool:
	def __init__(self, pool_size=50):
		self.pool_size = pool_size
		self.images = []

	def query(self, image):
		if self.pool_size == 0:
			return image

		if len(self.images) < self.pool_size:
			self.images.append(image)
			return image
		else:
			p = np.random.rand()
			if p > 0.5:
				idx = np.random.randint(0, self.pool_size)
				tmp = self.images[idx].copy()
				self.images[idx] = image.copy()
				return tmp
			else:
				return image
