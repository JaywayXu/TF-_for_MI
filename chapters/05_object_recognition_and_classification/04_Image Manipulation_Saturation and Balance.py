"""饱和与平衡
tensorflow实现了一些通过修改饱和度,色调,对比度和亮度的函数
利用这些函数可以对这些图像的属性进行操作和随机修改.
对属性进行修改能够使CNN精确匹配经过编辑的或不同光照条件下的图像的特征"""
import tensorflow as tf

example_red_pixel = tf.constant([254., 2., 15.])
adjust_brightness = tf.image.adjust_brightness(example_red_pixel, 0.2)
# 调整亮度,其实也就是将像素点的所有通道值中的数都变加上delta
sess = tf.Session()
print(sess.run(adjust_brightness))
# [ 254.19999695    2.20000005   15.19999981]
# 返回一个和原来图片shape一样的图片

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

"""调节图片对比度,图片对比度降低则将会生成一个识别度相当差的新图像.
最好选择一个较小的增量,过大的增量会导致图片饱和,像素会呈现全黑或全白的情况"""
adjust_contrast = tf.image.adjust_contrast(image, -.5)

print("调节图片的对比度减少0.5", sess.run(tf.slice(adjust_contrast, [1, 0, 0], [1, 3, 3])))
# 调节图片的对比度减少0.5
# [[[169  76 125]
#   [171 126  13]
#   [167  71 122]]]

"""改变图片的色度,使色彩更加丰富,该调整函数接受一个delta参数,用于控制需要调节的色度数量"""
adjust_hue = tf.image.adjust_hue(image, 0.7)

print("调节图片色度增加0.7", sess.run(tf.slice(adjust_hue, [1, 0, 0], [1, 3, 3])))
# 调节图片色度增加0.7
# [[[195  38   5]
#   [ 49 227   6]
#   [205  46  11]]]
