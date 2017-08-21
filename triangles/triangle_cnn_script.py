import numpy as np 
import matplotlib.pyplot as plt
import sys
import math
import scipy as sp
from PIL import Image, ImageDraw
import tensorflow as tf 
import os
import random 
folder_loc = '/media/pawan/0B6F079E0B6F079E/PYTHON_SCRIPTS/my_libs/'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(folder_loc),os.path.pardir)))

import my_libs.deep_learning_fns as dp

###########################################################################



images_loc = 'images/'
images_list = os.listdir(images_loc)
train_size = 4000
test_size = 100
images_train = images_list[0:train_size]
images_test= images_list[train_size:train_size+test_size]


###########################################################################


images = []

for i in range(0,train_size):
    image = plt.imread(images_loc+str(i)+'.png')
    image = np.array(image)
    image = image[:,:,0]
    
    images.append(image)
    
    
images = np.array(images)
images = images.reshape(images.shape[0], images.shape[1], images.shape[1],1)
images.shape



###########################################################################




images_test =[]
for i in range(train_size,train_size+test_size):
    image = plt.imread(images_loc+str(i)+'.png')
    image = np.array(image)
    image = image[:,:,0]
    
    images_test.append(image)
    
    
images_test = np.array(images_test)
images_test = images_test.reshape(images_test.shape[0], images_test.shape[1], images_test.shape[1],1)
images_test.shape

###########################################################################


angles = np.loadtxt('angles')
scales = np.loadtxt('scales')

train_angles = angles[0:train_size]
test_angles = angles[train_size:train_size+test_size]

train_scales = scales[0:train_size]
test_scales = scales[train_size: train_size+test_size]

# train_y = np.array([list(train_angles), list(train_scales)])
train_y = np.array([list(train_angles)])
train_y = train_y.reshape(train_y.shape[1], train_y.shape[0])


train_y.shape

###########################################################################




test_y = np.array([list(test_angles)])
test_y = test_y.reshape(test_y.shape[1], test_y.shape[0])
test_y.shape








###########################################################################






img_size = 50
img_flat  = img_size*img_size 
 

full_conn1 = 128 
full_conn2 = 32
full_connFinal = 1













###########################################################################

x = tf.placeholder(tf.float32, shape=[None, img_flat])
x_image = tf.reshape(x, [-1,img_size,img_size,1])
y = tf.placeholder(tf.float32, shape=[None, 1])




###########################################################################









num_channels= 1 

num_filters_1 = 16 
filter_size_1 = 3 

num_filters_2 = 32
filter_size_2 = 3

num_filters_3 = 64
filter_size_3 = 3

num_filters_4 = 128 
filter_size_4 = 3








###########################################################################



layer_conv_1,weights = dp.conv2d(input=x_image,
                              filter_size=filter_size_1,
                              num_channels=num_channels,
                              num_filters=num_filters_1,
                              use_pool=True)

layer_conv_2,weights_2 = dp.conv2d(input=layer_conv_1,
                              filter_size=filter_size_1,
                              num_channels=num_filters_1,
                              num_filters=num_filters_2,
                              use_pool=True)

layer_conv_3,weights_3 = dp.conv2d(input=layer_conv_2,
                              filter_size=filter_size_2,
                              num_channels=num_filters_2,
                              num_filters=num_filters_3,
                              use_pool=True)


layer_conv_4,weights_4 = dp.conv2d(input=layer_conv_3,
                              filter_size=filter_size_1,
                              num_channels=num_filters_3,
                              num_filters=num_filters_4,
                              use_pool=True)


# layer_conv_5,weights_5 = dp.conv2d(input=layer_conv_4,
#                               filter_size=filter_size_2,
#                               num_channels=num_filters_3,
#                               num_filters=num_filters_3,
#                               use_pool=False)



# layer_conv_6,weights_6 = dp.conv2d(input=layer_conv_5,
#                               filter_size=filter_size_2,
#                               num_channels=num_filters_3,
#                               num_filters=num_filters_3,
#                               use_pool=False)

# layer_conv_7,weights_7 = dp.conv2d(input=layer_conv_6,
#                               filter_size=filter_size_2,
#                               num_channels=num_filters_3,
#                               num_filters=num_filters_4,
#                               use_pool=True)



# layer_conv_7






###########################################################################

flat_1, num_features_f =  dp.flatten_layer(layer_conv_4)
full_conn_2 = dp.conn_layer(flat_1,num_features_f, full_conn1,relu=False)
full_conn_3 = dp.conn_layer(full_conn_2,full_conn1, full_conn2,relu=False)
full_conn_result = dp.conn_layer(full_conn_3,full_conn2,full_connFinal, relu=True)
full_conn_result







###########################################################################
learning_rate = 4.0

cost_function =(tf.reduce_sum(tf.square(full_conn_result-y)))/train_size
optimizer = tf.train.AdadeltaOptimizer(learning_rate).minimize(cost_function)




###########################################################################


def get_batch(batch_size):

    images_list =list(images)
    train_y_list =list(train_y)

    range_list= range(0,train_size)
    batch = random.sample(range_list,batch_size)
    
    images_batch = images[batch]
    train_y_batch = train_y[batch]

    return images_batch, train_y_batch





init = tf.global_variables_initializer()



###########################################################################

sess= tf.Session()
sess.run(init)

batch_size=  64

total_num_images = train_size
total_runs = int(total_num_images/batch_size)



epochs= 70
for epoch in range(0,epochs):
    
    for i in range(0,total_runs): 
        images_batch, train_y_batch = get_batch(batch_size)
        feed_dict_train = {x_image:images_batch,y:train_y_batch}
        sess.run(optimizer,feed_dict = feed_dict_train)
        cost_value = sess.run(cost_function, feed_dict = feed_dict_train)

#         if(i%20 == 0 ):
#             print("batch number:", i)
#             print("cost value :", cost_value)
    
    # images_batch, train_y_batch = get_batch(batch_size)
    feed_dict_train = {x_image:images,y:train_y}
    cost_value = sess.run(cost_function, feed_dict = feed_dict_train)
    print("Epoch:", epoch, "End of Epoch cost value :", cost_value)
         








###########################################################################



feed_dict_test  = {x_image:images_test}
test_values = sess.run(full_conn_result,feed_dict = feed_dict_test)
diff_values = np.abs(test_values-test_y)
mean_value = np.mean(diff_values)

print("mean value :", mean_value)