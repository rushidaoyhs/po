import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
print(mnist.train.next_batch(50)[0].shape)
print(mnist.train.next_batch(50)[1].shape)
print(mnist.train.images.shape)


