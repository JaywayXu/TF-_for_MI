import tensorflow as tf

"""convolution2d层与tf.nn.conv2d逻辑相同,但还包括权值初始化,偏置初始化,
可训练的变量输出,偏置相加,以及添加激活函数的功能"""
image_input = tf.constant([
    [
        [[0., 0., 0.], [255., 255., 255.], [254., 0., 0.]],
        [[0., 191., 0.], [3., 108., 233.], [0., 191., 0.]],
        [[254., 0., 0.], [255., 255., 255.], [0., 0., 0.]]
    ]
])
# shape(1,3,3,3)
conv2d = tf.contrib.layers.convolution2d(
    image_input,
    num_outputs=4,
    kernel_size=(1,1),          # 这里表示滤波器的高度和宽度
    activation_fn=tf.nn.relu,
    stride=(1, 1),              # 对image_batch和imput_channels的跨度值
    trainable=True)


# It's required to initialize the variables used in convolution2d's setup.

sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(tf.shape(image_input)))
print(sess.run(conv2d))
print(sess.run(tf.shape(conv2d)))
#   [1 3 3 3]
# [[[[   0.            0.            0.            0.        ]
#    [   3.33509827    0.            0.           91.75085449]
#    [   0.          175.44801331    0.            0.        ]]
#
#   [[   0.            0.            0.           24.43800926]
#    [  40.99788666    0.            0.          219.45314026]
#    [   0.            0.            0.           24.43800926]]
#
#   [[   0.          175.44801331    0.            0.        ]
#    [   3.33509827    0.            0.           91.75085449]
#    [   0.            0.            0.            0.        ]]]]
#   [1 3 3 4]
