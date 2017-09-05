import tensorflow as tf

input_batch = tf.constant([
    [  # First Input
        [[0.0], [1.0]],
        [[2.0], [3.0]]
    ],
    [  # Second Input
        [[2.0], [4.0]],
        [[6.0], [8.0]]
    ]
])

kernel = tf.constant([
    [
        [[1.0, 2.0]]
    ]
])
conv2d = tf.nn.conv2d(input_batch, kernel, strides=[1, 1, 1, 1], padding='SAME')
with tf.Session() as sess:
    print(sess.run(tf.shape(input_batch)))
    # [2 2 2 1]
    print(sess.run(tf.shape(kernel)))
    # [1 1 1 2]
    print(sess.run(conv2d))
    # [
    # [[[0.   0.]
    #    [1.   2.]]
    #
    #  [[2.   4.]
    #     [3.   6.]]]
    #
    #
    # [[[2.   4.]
    #   [4.   8.]]
    #
    # [[6.  12.]
    # [8.
    # 16.]]]]
    print(sess.run(tf.shape(conv2d)))
    # [2 2 2 2]

    """输出是另一个1input_batch同秩的张量,但是其最内层维度与卷积核相同
    输出层与输入层在最内的像素点级别有差异"""
    print(sess.run(input_batch)[0][1][1])  # 表示输入图中右下角的像素点
    print(sess.run(conv2d)[0][1][1])  # 表示输出图中右下角的像素点
    # [ 3.]
    # [3.   6.]
