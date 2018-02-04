import tensorflow as tf
import matplotlib as mil

# mil.use('svg')
mil.use("nbagg")
from matplotlib import pyplot

fig = pyplot.gcf()  # 获取当前图像
fig.set_size_inches(4, 4)  # 设置当前图像大小

image_filename = "./images/chapter-05-object-recognition-and-classification/convolution/n02113023_219.jpg"
filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once(image_filename))

image_reader = tf.WholeFileReader()
_, image_file = image_reader.read(filename_queue)  # 读取文件名队列
image = tf.image.decode_jpeg(image_file)  # 图片的值
sess = tf.Session()
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())  # 初始化变量
sess.run(init_op)
coord = tf.train.Coordinator()  # 初始化线程控制器
threads = tf.train.start_queue_runners(coord=coord, sess=sess)  # 初始化线程

image_batch = tf.image.convert_image_dtype(tf.expand_dims(image, 0), tf.float32, saturate=False)
"""
tf.expand_dims(input, dim, name = None)
解释：这个函数的作用是向input中插入维度是1的张量。
我们可以指定插入的位置dim，dim的索引从0开始，dim的值也可以是负数，从尾部开始插入，符合 python 的语法。
这个操作是非常有用的。举个例子，如果你有一张图片，数据维度是[height, width, channels]，你想要加入“批量”这个信息，
那么你可以这样操作expand_dims(images, 0)，那么该图片的维度就变成了[1, height, width, channels]。"""
kernel_3 = tf.constant([
    [
        [[-1., 0., 0.], [0., -1., 0.], [0., 0., -1.]],
        [[-1., 0., 0.], [0., -1., 0.], [0., 0., -1.]],
        [[-1., 0., 0.], [0., -1., 0.], [0., 0., -1.]]
    ],
    [
        [[-1., 0., 0.], [0., -1., 0.], [0., 0., -1.]],
        [[8., 0., 0.], [0., 8., 0.], [0., 0., 8.]],
        [[-1., 0., 0.], [0., -1., 0.], [0., 0., -1.]]
    ],
    [
        [[-1., 0., 0.], [0., -1., 0.], [0., 0., -1.]],
        [[-1., 0., 0.], [0., -1., 0.], [0., 0., -1.]],
        [[-1., 0., 0.], [0., -1., 0.], [0., 0., -1.]]
    ]
])
# kernel_3.shape(3, 3, 3, 3)此时利用的是三个卷积核
conv2d_3 = tf.nn.conv2d(image_batch, kernel_3, [1, 1, 1, 1], padding="SAME")
print("The shape of image of conv2d_3 is ", sess.run(tf.shape(conv2d_3)))
# 此时选择两个相同大小的卷积核
kernel_2 = tf.constant([
    [
        [[-1., 0., 0.], [0., -1., 0.], [0., 0., -1.]],
        [[-1., 0., 0.], [0., -1., 0.], [0., 0., -1.]],
        [[-1., 0., 0.], [0., -1., 0.], [0., 0., -1.]]
    ],
    [
        [[-1., 0., 0.], [0., -1., 0.], [0., 0., -1.]],
        [[8., 0., 0.], [0., 8., 0.], [0., 0., 8.]],
        [[-1., 0., 0.], [0., -1., 0.], [0., 0., -1.]]
    ],
])
conv2d_2 = tf.nn.conv2d(image_batch, kernel_2, [1, 1, 1, 1], padding="SAME")
print("The shape of image of conv2d_2 is ", sess.run(tf.shape(conv2d_2)))
# 选择四个相同大小的卷积核
kernel_4 = tf.constant([
    [
        [[-1., 0., 0.], [0., -1., 0.], [0., 0., -1.]],
        [[-1., 0., 0.], [0., -1., 0.], [0., 0., -1.]],
        [[-1., 0., 0.], [0., -1., 0.], [0., 0., -1.]]
    ],
    [
        [[-1., 0., 0.], [0., -1., 0.], [0., 0., -1.]],
        [[8., 0., 0.], [0., 8., 0.], [0., 0., 8.]],
        [[-1., 0., 0.], [0., -1., 0.], [0., 0., -1.]]
    ],
    [
        [[-1., 0., 0.], [0., -1., 0.], [0., 0., -1.]],
        [[-1., 0., 0.], [0., -1., 0.], [0., 0., -1.]],
        [[-1., 0., 0.], [0., -1., 0.], [0., 0., -1.]]
    ],
    [
        [[-1., 0., 0.], [0., -1., 0.], [0., 0., -1.]],
        [[8., 0., 0.], [0., 8., 0.], [0., 0., 8.]],
        [[-1., 0., 0.], [0., -1., 0.], [0., 0., -1.]]
    ]
])
conv2d_4 = tf.nn.conv2d(image_batch, kernel_4, [1, 1, 1, 1], padding="SAME")
print("The shape of image of conv2d_4 is ", sess.run(tf.shape(conv2d_4)))
activation_map = sess.run(tf.minimum(tf.nn.relu(conv2d_3), 255))
# 调用tf.minimum和tf.nn.relu的目的是将卷积值保持存在RGB颜色值的合成范围在[0,255]内
# relu函数x<0时值为0,x>0时可以有无穷大tf.minimum函数的支持广播模式,返回在x,y两个量中较小的值.

pyplot.imshow(activation_map[0])
# 保存当前图片
fig.savefig("./images/chapter-05-object-recognition-and-classification/convolution/example-edge-detection.png")
ath_kernel = tf.constant([
    [
        [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
        [[-1., 0., 0.], [0., -1., 0.], [0., 0., -1.]],
        [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]
    ],
    [
        [[-1., 0., 0.], [0., -1., 0.], [0., 0., -1.]],
        [[5., 0., 0.], [0., 5., 0.], [0., 0., 5.]],
        [[-1., 0., 0.], [0., -1., 0.], [0., 0., -1.]]
    ],
    [
        [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
        [[-1., 0., 0.], [0., -1., 0.], [0., 0., -1.]],
        [[0, 0., 0.], [0., 0., 0.], [0., 0., 0.]]
    ]
])

ath_conv2d = tf.nn.conv2d(image_batch, ath_kernel, [1, 1, 1, 1], padding="SAME")
activation_map = sess.run(tf.minimum(tf.nn.relu(ath_conv2d), 255))
fig = pyplot.gcf()
pyplot.imshow(activation_map[0])
fig.set_size_inches(4, 4)
fig.savefig("./images/chapter-05-object-recognition-and-classification/convolution/example-sharpen.png")
coord.request_stop()  # 关闭线程控制器
coord.join(threads)  # 关闭线程
sess.close()  # 关闭回话
