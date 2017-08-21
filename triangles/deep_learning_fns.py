from __future__ import division
import numpy as np
import tensorflow as tf

from numpy import ogrid, repeat, newaxis
import matplotlib.pyplot as plt
from skimage import io


def get_kernel_size(factor):
    """
    Find the kernel size given the desired factor of upsampling.
    """
    return 2 * factor - factor % 2


def upsample_filt(size):
    """
    Make a 2D bilinear kernel suitable for upsampling of the given (h, w) size.
    """
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)


def bilinear_upsample_weights(factor, number_of_classes):
    """
    Create weights matrix for transposed convolution with bilinear filter
    initialization.
    """
    
    filter_size = get_kernel_size(factor)
    
    weights = np.zeros((filter_size,
                        filter_size,
                        number_of_classes,
                        number_of_classes), dtype=np.float32)
    
    upsample_kernel = upsample_filt(filter_size)
    
    for i in range(0,number_of_classes):
        
        weights[:, :, i, i] = upsample_kernel
    
    return weights


def upsample_tf(factor, input_img):
    
    number_of_classes = input_img.shape[2]
    
    new_height = input_img.shape[0] * factor
    new_width = input_img.shape[1] * factor
    
    expanded_img = np.expand_dims(input_img, axis=0)

    with tf.Graph().as_default():
        with tf.Session() as sess:
            with tf.device("/cpu:0"):

                upsample_filt_pl = tf.placeholder(tf.float32)
                logits_pl = tf.placeholder(tf.float32)

                upsample_filter_np = bilinear_upsample_weights(factor,
                                        number_of_classes)

                res = tf.nn.conv2d_transpose(logits_pl, upsample_filt_pl,
                        output_shape=[1, new_height, new_width, number_of_classes],
                        strides=[1, factor, factor, 1])

                final_result = sess.run(res,
                                feed_dict={upsample_filt_pl: upsample_filter_np,
                                           logits_pl: expanded_img})
    
    return final_result.squeeze()




def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.05)
  return tf.Variable(initial)


# Get the 
def bias_variable(length):
  initial = tf.constant(0.05, shape=[length])
  return tf.Variable(initial)



def conv2d(input,filter_size,num_channels,num_filters, use_pool=True):
    
    shape = [filter_size,filter_size,num_channels,num_filters  ]
    W = weight_variable(shape=shape)
    bias = bias_variable(length =num_filters)
    
    layer =  tf.nn.conv2d(input=input, filter =W, strides=[1, 1, 1, 1], padding='SAME')
  
    layer +=bias 
    if use_pool: 
        layer = tf.nn.max_pool(value=layer, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
 
    layer = tf.nn.relu(layer)
    return layer,W 




def flatten_layer(layer): 
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer_flat = tf.reshape(layer, [-1, num_features])
    return layer_flat, num_features



def conn_layer(input,num_inputs, num_outputs, relu=True):

    W  = weight_variable([num_inputs,num_outputs])
    bias = bias_variable(num_outputs)
    layer = tf.matmul(input, W) + bias
    if relu: 
        layer = tf.nn.relu(layer)
        
    return(layer)