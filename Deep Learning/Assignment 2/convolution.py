from __future__ import absolute_import

import os
import tensorflow as tf
import numpy as np
import random
import math

def conv2d(inputs, filters, strides, padding):
	"""
	Performs 2D convolution given 4D inputs and filter Tensors.
	:param inputs: tensor with shape [num_examples, in_height, in_width, in_channels]
	:param filters: tensor with shape [filter_height, filter_width, in_channels, out_channels]
	:param strides: MUST BE [1, 1, 1, 1] - list of strides, with each stride corresponding to each dimension in input
	:param padding: either "SAME" or "VALID", capitalization matters
	:return: outputs, NumPy array or Tensor with shape [num_examples, output_height, output_width, output_channels]
	"""

	assert inputs.shape[3] == filters.shape[2], "inputs's number of 'in channels' is not equal to filters's number of 'in channels"
	
	num_examples = inputs.shape[0]
	in_height = inputs.shape[1]
	in_width = inputs.shape[2]
	input_in_channels = inputs.shape[3]

	filter_height = filters.shape[0]
	filter_width = filters.shape[1]
	filter_in_channels = filters.shape[2]
	filter_out_channels = filters.shape[3]

	num_examples_stride = strides[0]
	strideY = strides[1]
	strideX = strides[2]
	channels_stride = strides[3]

	# Cleaning padding input
	if padding == "SAME":
		pad_Y = max((in_height - 1) * strideY + filter_height - in_height, 0)
		pad_X =  max((in_width - 1) * strideX + filter_width - in_width, 0)
		pad_top = pad_Y // 2
		pad_bottom = pad_Y - pad_top
		pad_left = pad_X // 2
		pad_right = pad_X - pad_left

	elif padding == "VALID":
		pad_top = pad_bottom = pad_left = pad_right = 0
	else:
		raise ValueError("Padding must be either 'SAME' or 'VALID'")
	
	padded_inputs = tf.pad(inputs, 
                       [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], 
                       mode='CONSTANT')
	
	conv = tf.nn.conv2d(padded_inputs, filters, strides, padding)
	return conv