import pickle as pickle
import numpy as np
import sys
import matplotlib.pyplot as plt
from matplotlib import cm


import time
t0=time.clock()
file = './safe_samples/safe_samples.pickle'
with open(file, 'rb') as handler:
    safe_samples = pickle.load(handler)
print('reading takes %f seconds' % (time.clock() - t0))
print(safe_samples['state'].shape)
print(np.unique(safe_samples['reward'], return_counts=True))

image_width, image_height = safe_samples['state'][0].shape
print('image_width, image_height=%d, %d' %(image_width, image_height))

#visualize data
# for i in range(safe_samples['state'].shape[0]):
#     plt.imshow(safe_samples['state'][i])
#     plt.title(str(safe_samples['reward'][i]))
#     plt.pause(3)
total_data = safe_samples['state'].shape[0]
VALIDATION_SIZE = int(0.3*total_data)

safe_samples['state'] = np.multiply(safe_samples['state'], 1.0/255.0)
validation_images = safe_samples['state'][:VALIDATION_SIZE]
validation_labels = safe_samples['reward'][:VALIDATION_SIZE]

train_images = safe_samples['state'][VALIDATION_SIZE:]
train_labels = safe_samples['reward'][VALIDATION_SIZE:]

def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot
train_y_hot = dense_to_one_hot(train_labels, np.unique(train_labels).shape[0])

file = 'train_images.pickle'
with open(file, 'wb') as handler:
    pickle.dump(train_images, handler)

file = 'train_labels.pickle'
with open(file, 'wb') as handler:
    pickle.dump(train_labels, handler)

file = 'train_y_hot.pickle'
with open(file, 'wb') as handler:
    pickle.dump(train_y_hot, handler)


print(train_y_hot[0:10])
print(train_labels[0:10])
print(train_images[0:10])
