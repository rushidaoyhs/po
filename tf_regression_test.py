import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import sys
print(sys.executable)

def regression(x_data, y_data, step_size):

    # model specification
    W= tf.Variable(tf.random_uniform([1], -1, 1))
    b = tf.Variable(tf.zeros([1]))
    y = W * x_data +b

    #objective function
    loss = tf.reduce_mean(tf.square(y -  y_data))
    optimizer = tf.train.GradientDescentOptimizer(step_size)
    train = optimizer.minimize(loss)

    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    err = []
    for tau in range(201):
        sess.run(train)
        err.append(sess.run(loss))
    return err

x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3
err = regression(x_data, y_data, 0.5)
err2 = regression(x_data, y_data, 0.1)
h1, = plt.plot(err, label = '0.5')
h2, = plt.plot(err2, label = '0.1')
plt.legend(handles = [h1, h2])
plt.show()