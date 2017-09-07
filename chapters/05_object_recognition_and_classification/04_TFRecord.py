# Reuse the image from earlier and give it a fake label
# 复用之前的图像，并赋予一个假标签
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
image_label = b'\x01'
# Assume the label data is in a one-hot representation (00000001)
# 假设标签数据位于一个独热的（one-hot）编码表示中，(00000001) 二进制8位'x01'
# Convert the tensor into bytes, notice that this will load the entire image file
# 将张量转换为字节型，注意这会加载整个图像文件。
image_loaded = sess.run(image)
image_bytes = image_loaded.tobytes()
image_height, image_width, image_channels = image_loaded.shape

# Export TFRecord
writer = tf.python_io.TFRecordWriter("./output/training-image.tfrecord")

# Don't store the width, height or image channels in this Example file to save space but not required.
example = tf.train.Example(features=tf.train.Features(feature={
    'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_label])),
    'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes]))
}))

# This will save the example to a text file tfrecord
writer.write(example.SerializeToString())
writer.close()
