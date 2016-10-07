import cPickle as pickle
import numpy as np
import sys
import matplotlib.pyplot as plt


print(sys.executable)
import tensorflow as tf
gamma = 0.9
image_width, image_height = 80, 80
file_header = 'samples_episodes/samples_random_policy_small.pickle'

# image number to output
IMAGE_TO_DISPLAY = 10

with open(file_header+'1', 'rb') as handle:
    samples = pickle.load(handle)

# visualize
rewards = samples['rewards']
# validate data
states = samples['states']


def display(img):
    print(img.shape)
    one_image = img.reshape(image_width, image_height)
    plt.axis('off')
    plt.imshow(one_image, cmap = 'Greys_r')

plt.figure(1)
display(states[0][0])


assert(len(states) == len(rewards))
image_size = (image_width, image_height)
total_data = 0
for epi in range(len(states)):
    assert(len(states[epi]) == len(rewards[epi]))
    assert(states[epi][0].shape == image_size)
    total_data += len(states[epi])
print('total data is %d' % total_data)

def play_an_episode(epi):
    plt.ion()
    plt.show()
    for step in range(len(states[epi])):
        one_image = states[epi][step]
        plt.axis('off')
        plt.title('step ' + str(step))
        plt.imshow(one_image, cmap='Greys_r')
        plt.pause(0.1)
for epi in range(5):
    play_an_episode(epi)


# reformat data into nd-array
train_images = np.zeros((total_data, image_width, image_height,), dtype = np.float32)
train_y_hot = np.zeros((total_data, 3), dtype = np.float32)
index = 0
rewards = np.zeros((total_data, 1), dtype = np.float32)
for epi in range(len(states)):
    for i in range(len(states[epi])):
        train_images[index] = np.multiply(states[epi][i][:, :, 0], 1.0 / 255.0)
        if rewards[epi][i] == -1:
            rewards[index] = 0.0
        elif rewards[epi][i] == 0:
            rewards[index] = 1.0
        else:
            rewards[index] = 2.0
        train_y_hot[index][int(rewards[index])] = 1.0
        index += 1

# plt.figure(2)
# display(x_all[0])
# plt.show()
print ('labels_flat[{0}] => {1}'.format(IMAGE_TO_DISPLAY, rewards[IMAGE_TO_DISPLAY]))
print(np.unique(rewards, return_counts = True))
labels_count = np.unique(rewards).shape[0]
print('labels_count => {0}'.format(labels_count))

# def dense_to
print(np.max(train_y_hot))
print(np.min(train_y_hot))
print(np.max(train_images))
print(np.min(train_images))
print(train_images.shape)
print(train_y_hot.shape)
print(rewards.shape)


import tensorflow as tf
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


W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x = tf.placeholder(tf.float32, shape=[None, 160, 160])
y_ = tf.placeholder(tf.float32, shape=[None, 1])

x_image = tf.reshape(x, [-1,image_width, image_height,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
print('h_conv1: ', h_conv1.get_shape())
h_pool1 = max_pool_2x2(h_conv1)
print('h_pool1: ', h_pool1.get_shape())

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
print('h_conv2: ', h_conv2.get_shape())
h_pool2 = max_pool_2x2(h_conv2)
print('h_pool2: ', h_pool2.get_shape())

h_pool2_flat = tf.reshape(h_pool2, [-1, 40 * 40 * 64])
W_fc1 = weight_variable([40 * 40 * 64, 1024])
b_fc1 = bias_variable([1024])

h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
print('h_fc1:', h_fc1.get_shape())

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
print('h_fc1_drop:', h_fc1_drop.get_shape())

W_fc2 = weight_variable([1024, 1])
b_fc2 = bias_variable([1])
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
print('y_conv:', y_conv.get_shape())

epochs_completed = 0
index_in_epoch = 0
num_examples = train_images.shape[0]
print('total number of samples is %d' % num_examples)
# serve data by batches
def next_batch(batch_size):
    global train_images
    global train_y_hot
    global rewards
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
        x_all = x_all[perm]
        y_all = y_all[perm]
        labels = labels[perm]
        # start next epoch
        start = 0
        index_in_epoch = batch_size
        assert batch_size <= num_examples
    end = index_in_epoch
    # print('start=%d, %d' % (start,end))
    return x_all[start:end], y_all[start:end], labels[start: end]

sess = tf.InteractiveSession()
# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
#cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv + 1e-8))
cross_entropy = tf.reduce_sum(tf.square(y_ - y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())

perm = np.arange(num_examples)
np.random.shuffle(perm)
train_images = train_images[perm]
train_y_hot = train_y_hot[perm]
rewards = rewards[perm]
W_conv1_diff = []
losses = []
y_conv_values = []
y_true = []
for i in range(10):
    x_batch, y_batch, label_batch = next_batch(50)
    print(np.unique(label_batch, return_counts=True))
    # train_accuracy = accuracy.eval(feed_dict={x: x_batch, y_: y_batch, keep_prob: 1.0})
    train_loss = cross_entropy.eval(feed_dict={x: x_batch, y_: label_batch, keep_prob: 1.0})
    losses.append(train_loss)
    y_conv_values.append(y_conv.eval(feed_dict={x: x_batch, y_: label_batch, keep_prob: 1.0}))
    y_true.append(label_batch)
    # print("step %d, training accuracy %g "%(i, train_accuracy))
    print('train_loss= ', train_loss)

    w_old = W_conv2.eval()
    sess.run(train_step, feed_dict={x: x_batch, y_: label_batch, keep_prob: 1.0})
    w_new = W_conv2.eval()
    err = np.sum(np.abs(w_old - w_new))

    print('w diff is %f' % err)
    W_conv1_diff.append(err)

plt.plot(np.vstack(np.asarray(y_true)), '-bx')
plt.plot(np.vstack(np.asarray(y_conv_values)), '-.ro')
plt.plot(losses)
plt.show()
BATCH_SIZE = 50
# predict = tf.argmax(y_conv, 1)
# predict_val = tf.reduce_max(y_conv, 1)
# predicted_labels = np.zeros(x_all.shape[0])
predicted_values = np.zeros(train_images.shape[0], dtype=np.float32)

for i in range(0,3): #x_all.shape[0]//BATCH_SIZE
    # predicted_labels[i*BATCH_SIZE : (i+1)*BATCH_SIZE] = predict.eval(feed_dict={x: x_all[i*BATCH_SIZE : (i+1)*BATCH_SIZE],
    #                                                                             keep_prob: 1.0})
    predicted_values[i*BATCH_SIZE : (i+1)*BATCH_SIZE] = y_conv.eval(feed_dict={x: train_images[i * BATCH_SIZE : (i + 1) * BATCH_SIZE],
                                                                                keep_prob: 1.0})

# print(W_conv1_diff)
# print(w_norm)

# output test image and prediction
display(train_images[IMAGE_TO_DISPLAY])
print ('predicted_labels[{0}] => {1}, value = {2}, correct lable => {3}'.format(IMAGE_TO_DISPLAY, predicted_labels[IMAGE_TO_DISPLAY], \
                                                                                predicted_values[IMAGE_TO_DISPLAY], \
                                                                                rewards[IMAGE_TO_DISPLAY]))

sess.close()
print('done')