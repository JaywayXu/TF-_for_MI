"""在大多数场景中,对图像的操作最好能在预处理阶段完成.预处理包括对图像裁剪,缩放以及灰度调整.
另一方面,在训练时对图像进行操作有一个重要的用例.当一副图像被加载后,可对其进行翻转或扭曲处理,
以使输入给网络的训练信息多样化.虽然这个步骤会进一步增加处理时间,但却有助于缓解过拟合现象"""
# 复用image图像.
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
print("从图像中心将10%抠出", sess.run(tf.image.central_crop(image, 0.1)))
# the image is:
#  [[[  0   9   4]
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
# 从图像中心将10%抠出 [[[  6  94 227]]]


"""裁剪通常在预处理阶段使用,但在训练阶段,若背景也有用时,可随机化裁剪区域起始位置到图像中心的偏移量来实现裁剪"""

# 这个裁剪方式进接受实值输入,即其仅能接受一个具有确定形状的张量,因此,输入图像需要事先在数据流图中运行.

real_image = sess.run(image)

bounding_crop = tf.image.crop_to_bounding_box(
    real_image, offset_height=0, offset_width=0, target_height=2, target_width=1)

print("The bounding_crop image is", sess.run(bounding_crop))

# The bounding_crop image is
# [[[  0   9   4]]
#
#  [[ 10 195   5]]]

# 这段代码意为从位于(0,0)的图像的左上角像素开始对图像裁剪.
