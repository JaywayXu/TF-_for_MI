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
print('the image is:', sess.run(image))
filename_queue.close(cancel_pending_enqueues=True)
coord.request_stop()
coord.join(threads)
image_label = b'\x01'
# Assume the label data is in a one-hot representation (00000001)
# 假设标签数据位于一个独热的（one-hot）编码表示中，(00000001) 二进制8位'x01'
# Convert the tensor into bytes, notice that this will load the entire image file
# 将张量转换为字节型，注意这会加载整个图像文件。
image_loaded = sess.run(image)
image_bytes = image_loaded.tobytes()  # 将张量转化为字节类型.
image_height, image_width, image_channels = image_loaded.shape

# Export TFRecord 导出TFRecord
writer = tf.python_io.TFRecordWriter("./output/training-image.tfrecord")

# Don't store the width, height or image channels in this Example file to save space but not required.
# 样本文件中不保存图像的宽度/高度和通道数,以便节省不要求分配的空间.
example = tf.train.Example(features=tf.train.Features(feature={
    'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_label])),
    'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes]))
}))

# This will save the example to a text file tfrecord
writer.write(example.SerializeToString())  # 序列化为字符串
writer.close()
# image = example.features.feature['image_bytes'].bytes_list.value
# label = example.features.feature['image_label'].int64_list.value
# 这样的方式进行读取.
"""标签的格式被称为独热编码(one-hot encoding)这是一种用于多类分类的有标签数据的常见的表示方法.
Stanford Dogs 数据集之所以被视为多类分类数据,是因为狗会被分类为单一品种,而非多个品种的混合,
在现实世界中,当预测狗的品种是,多标签解决方案通常较为有效,因为他们能够同时匹配属于多个品种的狗"""

"""
这段代码中,图像被加载到内存中并被转换为字节数组
image_bytes = image_loaded.tobytes() 
然后通过tf.train.Example函数将values和labels以value的方式加载到example中,
example = tf.train.Example(features=tf.train.Features(feature={
    'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_label])),
    'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes]))
}))
example在写入保存到磁盘之前需要先通过SerializeToString()方法将其序列化为二进制字符串.
序列化是一种将内存对象转化为可安全传输到某种文件的格式.
上面序列化的样本现在被保存为一种可被加载的格式,并可被反序列化为这里的样本格式

由于图像被保存为TFRecord文件,可以被再次从TFRecord文件加载.这样比将图像及其标签分开加载会节省一些时间
"""
# Load TFRecord
# 加载TFRecord文件,获取文件名队列
tf_record_filename_queue = tf.train.string_input_producer(["./output/training-image.tfrecord"])

# Notice the different record reader, this one is designed to work with TFRecord files which may
# have more than one example in them.
# 注意这个不同的记录读取其,它的设计意图是能够使用可能会包含多个样本的TFRecord文件
tf_record_reader = tf.TFRecordReader()
_, tf_record_serialized = tf_record_reader.read(tf_record_filename_queue)
# 通过阅读器读取value值,并保存为tf_record_serialized

# The label and image are stored as bytes but could be stored as int64 or float64 values in a
# serialized tf.Example protobuf.
# 标签和图像都按字节存储,但也可按int64或float64类型存储于序列化的tf.Example protobuf文件中
tf_record_features = tf.parse_single_example(  # 这是一个模板化的东西,大部分都是这么写的
    tf_record_serialized,
    features={
        'label': tf.FixedLenFeature([], tf.string),
        'image': tf.FixedLenFeature([], tf.string),
    })
"""
class FixedLenFeature(collections.namedtuple(
    "FixedLenFeature", ["shape", "dtype", "default_value"])):"""

"""Configuration for parsing a fixed-length input feature.
    用于解析固定长度的输入特性的配置。
  To treat sparse input as dense, provide a `default_value`; otherwise,
  the parse functions will fail on any examples missing this feature.
把稀疏的输入看作是稠密的，提供一个默认值;否则，解析函数将缺少属性值的情况下报错。
  Fields:
    shape: Shape of input data.输入数据的形状
    dtype: Data type of input.输入数据类型
    default_value: Value to be used if an example is missing this feature. It
        must be compatible with `dtype` and of the specified `shape`.
    如果一个示例缺少属性值，那么将使用该默认值。它必须与dtype和指定的形状兼容。
"""
# 但是在实际使用的过程中这里的features的是根据原先的保存时的名字对应的,而数据类型可以自行选取.

# Using tf.uint8 because all of the channel information is between 0-255
# 使用tf.uint8类型,因为所有的通道信息都处于0~255的范围内
tf_record_image = tf.decode_raw(
    tf_record_features['image'], tf.uint8)
# tf.decode_raw()函数将将字符串的字节重新解释为一个数字的向量。

# Reshape the image to look like the image saved, not required
# 调整图像的尺寸,使其与保存的图像类似,但这并不是必需的
tf_record_image = tf.reshape(
    tf_record_image,
    [image_height, image_width, image_channels])
# Use real values for the height, width and channels of the image because it's required
# 用是指表示图像的高度,宽度和通道,因为必须对输入的形状进行调整
# to reshape the input.

tf_record_label = tf.cast(tf_record_features['label'], tf.string)
sess.close()
sess = tf.InteractiveSession()
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
sess.run(init_op)
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord, sess=sess)
print("equal the image before and now", sess.run(tf.equal(image, tf_record_image)))  # 检查原始图像和加载后的图像是否一致
"""首先,按照与其他文件相同的方式加载该文件,主要区别在于该文件主要有TFRecordReaader对象读取.
tf.parse_single_example对TFRecord进行解析,然后图像按原始字节(tf.decode_raw)进行读取"""
print("The lable of the image:", sess.run(tf_record_label))  # 输出图像的标签
tf_record_filename_queue.close(cancel_pending_enqueues=True)
coord.request_stop()
coord.join(threads)
