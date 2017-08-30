import tensorflow as tf
import numpy as np

sess = tf.Session()
image_batch = tf.constant([
    [  # First Image
        [[0, 255, 0], [0, 255, 0], [0, 255, 0]],
        [[0, 255, 0], [0, 255, 0], [0, 255, 0]]
    ],
    [  # Second Image
        [[0, 0, 255], [0, 0, 255], [0, 0, 255]],
        [[0, 0, 255], [0, 0, 255], [0, 0, 255]]
    ]
])
getshape = tf.shape(image_batch)
print(sess.run(getshape))
# [2 2 3 3]
# 输出第一组维度表明了图像数量,第2组维度对应图像的高度,
# 第3个维度表明图像的宽度,颜色通道数量即RGB值对应于最后一个维度.
# 所以要访问第一个图像的第一个像素点即为
print(sess.run(image_batch)[0][0][0])
# 因为第一个像素点位于宽和高都是第0个点的位置.[  0 255   0]
