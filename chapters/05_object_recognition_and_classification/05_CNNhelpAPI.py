"""主要测试tf.where的使用"""
import tensorflow as tf
import numpy as np

a = np.array([[1], [2], [3]])
b = np.array([[1], [7], [8], [4], [5], [2], [3]])
c = tf.map_fn(lambda l: tf.where(tf.equal(b, l))[0][0], a, dtype=tf.int64)

# c = tf.where(tf.equal(a, b))
# Dimensions must be equal, but are 3 and 7 for 'Equal' (op: 'Equal') with input shapes: [3,1], [7,1].
print(a)
print(b)

sess = tf.Session()
print(sess.run(c))
# [0 5 6]