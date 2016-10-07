import pickle as pickle
import numpy as np
import sys
import matplotlib.pyplot as plt
from matplotlib import cm
import tensorflow as tf

file = 'train_images.pickle'
with open(file, 'rb') as handler:
    train_images = pickle.load(handler)

file = 'train_y_hot.pickle'
with open(file, 'rb') as handler:
    train_y_hot = pickle.load(handler)

epochs_completed = 0
index_in_epoch = 0
num_examples = train_images.shape[0]
print('total number of samples is %d' % num_examples)
# serve data by batches
def next_batch(batch_size):
    global train_images
    global train_y_hot
    global index_in_epoch
    global epochs_completed

    start = index_in_epoch
    index_in_epoch += batch_size

    # when all trainig data have been already used, it is reorder randomly
    if index_in_epoch > num_examples:
        # print('finished epoch, shuffle the data')
        epochs_completed += 1
        perm = np.arange(num_examples)
        np.random.shuffle(perm)
        train_images = train_images[perm]
        train_y_hot = train_y_hot[perm]
        # start next epoch
        start = 0
        index_in_epoch = batch_size
        assert batch_size <= num_examples
    end = index_in_epoch
    # print('start=%d, %d' % (start,end))
    return train_images[start:end], train_y_hot[start:end]

image_width, image_height = train_images[0].shape
print('image_width, image_height=%d, %d' %(image_width, image_height))

LEARNING_RATE = 1e-2
## multi-layer
def weight_variable(shape):
  # initial = tf.truncated_normal(shape, stddev=0.1)
  initial = tf.truncated_normal(shape, mean = 0.1, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


W_conv1 = weight_variable([2, 2, 1, 32])
b_conv1 = bias_variable([32])

x = tf.placeholder(tf.float32, shape=[None, image_width, image_height])
y_ = tf.placeholder(tf.float32, shape=[None, 3])

x_image = tf.reshape(x, [-1,image_width, image_height,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
print('h_conv1: ', h_conv1.get_shape())
h_pool1 = max_pool_2x2(h_conv1)
print('h_pool1: ', h_pool1.get_shape())
# display 32 fetures in 4 by 8 grid
layer1 = tf.reshape(h_conv1, (-1, image_height, image_width, 4 ,8))
# reorder so the channels are in the first dimension, x and y follow.
layer1 = tf.transpose(layer1, (0, 3, 1, 4,2))
layer1 = tf.reshape(layer1, (-1, image_height*4, image_width*8))

W_conv2 = weight_variable([2, 2, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
print('h_conv2: ', h_conv2.get_shape())
h_pool2 = max_pool_2x2(h_conv2)
print('h_pool2: ', h_pool2.get_shape())

# display 64 fetures in 4 by 16 grid
layer2 = tf.reshape(h_conv2, (-1, 40, 40, 4 ,16))
# reorder so the channels are in the first dimension, x and y follow.
layer2 = tf.transpose(layer2, (0, 3, 1, 4,2))
layer2 = tf.reshape(layer2, (-1, 40*4, 40*16))

h_pool2_flat = tf.reshape(h_pool2, [-1, 20 * 20 * 64])
W_fc1 = weight_variable([20 * 20 * 64, 1024])
b_fc1 = bias_variable([1024])

h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
print('h_fc1:', h_fc1.get_shape())

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
print('h_fc1_drop:', h_fc1_drop.get_shape())

W_fc2 = weight_variable([1024, 3])
b_fc2 = bias_variable([3])
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
print('y_conv:', y_conv.get_shape())


sess = tf.InteractiveSession()
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())

perm = np.arange(num_examples)
np.random.shuffle(perm)
train_images = train_images[perm]
train_y_hot = train_y_hot[perm]
W_conv1_diff = []
losses = []
y_conv_values = []
y_true = []
for i in range(100):
    x_batch, y_batch = next_batch(50)
    # train_loss = cross_entropy.eval(feed_dict={x: x_batch, y_: y_batch, keep_prob: 1.0})

    # w_old = W_conv2.eval()
    sess.run(train_step, feed_dict={x: x_batch, y_: y_batch, keep_prob: 1.0})
    # w_new = W_conv2.eval()
    # err = np.sum(np.abs(w_old - w_new))
    train_accuracy = accuracy.eval(feed_dict={x: x_batch, y_: y_batch, keep_prob: 1.0})
    # print('train_loss= ', train_loss)
    print('i=%d train_accuracy=%f' % (i, train_accuracy))
    losses.append(train_accuracy)
    y_conv_values.append(y_conv.eval(feed_dict={x: x_batch, y_: y_batch, keep_prob: 1.0}))
    # print('w diff is %f' % err)
    # W_conv1_diff.append(err)

IMAGE_TO_DISPLAY = 10
layer1_grid = layer1.eval(feed_dict={x: train_images[IMAGE_TO_DISPLAY:IMAGE_TO_DISPLAY+1], keep_prob: 1.0})
plt.axis('off')
plt.imshow(layer1_grid[0], cmap=cm.seismic )

plt.figure(2)
layer2_grid = layer2.eval(feed_dict={x: train_images[IMAGE_TO_DISPLAY:IMAGE_TO_DISPLAY+1], keep_prob: 1.0})
plt.axis('off')
plt.imshow(layer2_grid[0], cmap=cm.seismic )

# plt.plot(np.vstack(np.asarray(y_true)), '-bx')
# plt.plot(np.vstack(np.asarray(y_conv_values)), '-.ro')
plt.plot(losses)
plt.show()