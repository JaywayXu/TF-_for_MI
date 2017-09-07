# tensorflow在设计师就医从磁盘快速加载文件作为目标,图像的加载与其他二进制文件加载相同,
# 只是图像的内容需要解码
# The match_filenames_once 将接受一个正则表达式,但在本例这是不需要的
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
print(sess.run(image))
filename_queue.close(cancel_pending_enqueues=True)
coord.request_stop()
coord.join(threads)
# [[[  0   9   4]
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
