"""当图片只有一种颜色时,我们称之为使用了灰度颜色空间,即单颜色通道
对于计算机视觉相关任务而言只需要轮廓无需借助大量的颜色信息,缩减颜色空间可以加速训练过程"""

import tensorflow as tf

image_filename = "./images/chapter-05-object-recognition-and-classification/working-with-images/test-input-image.jpg"
# 获得文件名列表
filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once(image_filename))
# 生成文件名队列
image_reader = tf.WholeFileReader()
_, image_file = image_reader.read(filename_queue)
# 通过阅读器返回一个键值对,其中value表示图像
image = tf.image.decode_jpeg(image_file)
# 通过tf.image.decode_jpeg解码函数对图片进行解码,得到图像.
sess = tf.Session()
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
sess.run(init_op)
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord, sess=sess)
print('the image is:', sess.run(image))
filename_queue.close(cancel_pending_enqueues=True)
coord.request_stop()
coord.join(threads)

"""灰度"""
print("IF this picture is not to be grayscale: ", sess.run(tf.slice(image, [0, 0, 0], [1, 3, 1])))
# the image is: [[[  0   9   4]
#   [254 255 250]
#   [255  11   8]]
#
#  [[ 10 195   5]
#   [  6  94 227]
#   [ 14 205  11]]
#
#  [[255  10   4]
#   [249 255 244]
#   [  3   8  11]]]
# IF this picture is not to be grayscale:
# [[[  0]
#   [254]
#   [255]]]
gray = tf.image.rgb_to_grayscale(image)
print("If this picture is to be gray:", sess.run(tf.slice(gray, [0, 0, 0], [1, 3, 1])))
# If this picture is to be gray: [[[  5]
#   [254]
#   [ 83]]]
"""这个例子将RGB图像转换为灰度图,tf.slice运算提取了最上一行的像素,并且查看其颜色值是否发生了变化"""

# HSV
# HSB空间,色度,饱和度,灰度值构成了HSV颜色空间结构和RGB模式十分相似,其中B表示亮度值

hsv = tf.image.rgb_to_hsv(tf.image.convert_image_dtype(image, tf.float32))
print("use HSV to express image", sess.run(hsv))
# use HSV to express image [[[ 0.40740743  1.          0.03529412]
#   [ 0.1999996   0.01960778  1.        ]
#   [ 0.00202429  0.96862745  1.        ]]
#
#  [[ 0.32894737  0.97435898  0.76470596]
#   [ 0.60030168  0.97356826  0.89019614]
#   [ 0.33075601  0.94634145  0.80392164]]
#
#  [[ 0.00398406  0.98431373  1.        ]
#   [ 0.25757566  0.04313719  1.        ]
#   [ 0.5625      0.72727269  0.04313726]]]


# RGB空间
# RGB空间,即是使用红绿蓝三原色来表示图片
"""tensorflow可以使用函数将其他颜色的表示形式转化为RGB形式"""

# 将hsv类型图片转化为rgb
rgb_hsv = tf.image.hsv_to_rgb(hsv)
# 将灰度值表示的图片转化为rgb
rgb_grayscale = tf.image.grayscale_to_rgb(gray)


"""图像数据类型转换"""
"""在这些例子中,普遍使用tf.to_float函数改变数据类型.
tensorflow还提供了一个内置函数,用于当图像数据类型发生变化是恰当的对像素值进行比例变换"""
# tf.image.convert_image_dtype(image, dtype, saturate=False)
