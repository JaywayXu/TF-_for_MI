"""主要测试tf.where的使用"""
import tensorflow as tf
import numpy as np


a = np.array([[5]])
a1 = np.array([[1], [2], [3]])
b = np.array([[1], [7], [8], [4], [5], [2], [3], [2], [3]])
# 对于[n,1]shape张量匹配必须使用map_fn函数，否则会出shape函数维度不匹配的错误
c1 = tf.map_fn(lambda l: tf.where(tf.equal(b, l))[0][0], a1, dtype=tf.int64)
c = tf.where(tf.equal(a, b))[0][0]

# c = tf.where(tf.equal(a1, b))[0][0] 这个语句就会出现下面维度不匹配的错误。
# Dimensions must be equal, but are 3 and 7 for 'Equal' (op: 'Equal') with input shapes: [3,1], [7,1].
sess = tf.Session()
print(sess.run(c))
print(sess.run(c1))

# 4
# [0 5 6]
