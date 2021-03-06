import re
import os

from PIL import Image as IMP
from util import *
from visualize import *

def gen_squares_and_save(img_path):
	img = load_image(img_path)

	xr = img.shape[0]
	yr = img.shape[1]

	window = 20 #n pixel squares taken
	counter = 0

	fname = re.match(r'^[\w\d\/]+\/([\w|\d|\-|_]+)\.[\w\d]+$', img_path).groups()[0]

	original = IMP.open(img_path)
	new_dir = 'data/gen_data/%s' % fname

	imgs_ret = []

	if not os.path.exists(new_dir):
		os.mkdir(new_dir)

		for ix in range(0, xr - window, window):
			for iy in range(0, yr - window, window):

				leny = window if iy + window < yr else yr - iy - 1
				lenx = window if ix + window < xr else xr - ix - 1

				cropped = original.crop((ix, iy, ix+lenx, iy+leny))
				i_path = 'data/gen_data/%s/%d.png' % (fname, counter)

				cropped.save(i_path)
				img_temp = load_image(i_path)
				imgs_ret.append(img_temp)
				counter+=1
	else:
		for f in os.listdir(new_dir):
			i_path = 'data/gen_data/%s/%s' % (fname, f)

			img_temp = load_image(i_path)
			imgs_ret.append(img_temp)

	return imgs_ret



