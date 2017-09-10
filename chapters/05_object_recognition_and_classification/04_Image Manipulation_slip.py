"""Tensorflow中有一些函数可以实现垂直翻转,水平翻转,随即翻转等操作可以防止模型对图像的翻转版本产生过拟合"""
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
"""
[[[  0   9   4]
  [254 255 250]
  [255  11   8]]

 [[ 10 195   5]
  [  6  94 227]
  [ 14 205  11]]

 [[255  10   4]
  [249 255 244]
  [  3   8  11]]]
"""
filename_queue.close(cancel_pending_enqueues=True)
coord.request_stop()
coord.join(threads)
top_left_pixels = tf.slice(image, [0, 0, 0], [2, 2, 3])
# 对图形取切片,其中从第[0,0,0]号像素点开始
# (从一开始)第1维取2个,第二维两个,第三维三个.

flip_horizon = tf.image.flip_left_right(top_left_pixels)
# 水平翻转
flip_vertical = tf.image.flip_up_down(flip_horizon)
# 垂直翻转
print("top_left_pixels is:", sess.run(top_left_pixels))
"""
 [[[  0   9   4]
  [254 255 250]]

 [[ 10 195   5]
  [  6  94 227]]]
"""
print("flip_vertical is:", sess.run(flip_vertical))
"""
 [[[  6  94 227]
  [ 10 195   5]]

 [[254 255 250]
  [  0   9   4]]]
"""

"""对图像进行随机翻转,即有可能是从左到右的翻转当然也有可能是从左到左的翻转,从左到左的翻转就是没有翻转.上下翻转同理"""

random_flip_horizon = tf.image.random_flip_left_right(top_left_pixels)
# 随机从左到右翻转
random_flip_vertical = tf.image.random_flip_up_down(random_flip_horizon)
# 随机从上到下翻转
print("random_flip_horizon", sess.run(random_flip_horizon))
print("random_flip_vertical", sess.run(random_flip_vertical))
# random_flip_horizon [[[  0   9   4]
#   [254 255 250]]
#
#  [[ 10 195   5]
#   [  6  94 227]]]
# random_flip_vertical [[[254 255 250]
#   [  0   9   4]]
#
#  [[  6  94 227]
#   [ 10 195   5]]]

"""将图片沿对角线翻转"""
transposeImg = tf.image.transpose_image(top_left_pixels)
print("transposeImg is", sess.run(transposeImg))
# transposeImg is [[[  0   9   4]
#   [ 10 195   5]]
#
#  [[254 255 250]
#   [  6  94 227]]]
