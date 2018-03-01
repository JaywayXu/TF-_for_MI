"""转换图像数据格式时需要将它们的颜色空间变为灰度空间,将图像尺寸修改为同一尺寸,并将标签依附于每幅图像"""
import tensorflow as tf

sess = tf.Session()
import glob

image_filenames = glob.glob("./imagenet-dogs/n02*/*.jpg")  # 访问imagenet-dogs文件夹中所有n02开头的子文件夹中所有的jpg文件

# image_filenames[0:2]  此语句表示image_filenames文件中的从第0个编号到第1个编号的值
# ['./imagenet-dogs\\n02085620-Chihuahua\\n02085620_10074.jpg',
# './imagenet-dogs\\n02085620-Chihuahua\\n02085620_10131.jpg']
# 此时image_filenames中保存的全部是类似于以上形式的值
# 注意书上的解释和这个输出和此处的输出与有很大的不同,原因是书本是用linux系统,
# 所以是以"/"对文件名进行分隔符的操作而此处不是windows下使用"\\"对文件名进行操作.

from itertools import groupby
from collections import defaultdict

training_dataset = defaultdict(list)  # 构造训练集集合，将traing_dateset设置为一个list对象，可以向其中添加成员
testing_dataset = defaultdict(list)  # 构造测试集集合，将testing_dateset设置为一个list对象，可以向其中添加成员

# Split up the filename into its breed and corresponding filename. The breed is found by taking the directory name
# 将文件名分解为品种和对应的文件名,品种对应于文件夹名称
image_filename_with_breed = map(lambda filename: (filename.split("/")[1].split("\\")[1], filename), image_filenames)
# 表示定义一个匿名函数lambda传入参数为filename,对filename以"/"为分隔符,然后取第二个值
# 再利用"\\"作为分隔符取第二个值,并且返回操作结果和filename
# 并且以image_filenames作为参数
# ('n02086646-Blenheim_spaniel', './imagenet-dogs\\n02086646-Blenheim_spaniel\\n02086646_3739.jpg')

# Group each image by the breed which is the 0th element in the tuple returned above
# 依据品种(上述返回的元组的第0个分量对元素进行分组)
for dog_breed, breed_images in groupby(image_filename_with_breed, lambda x: x[0]):
    # Enumerate each breed's image and send ~20% of the images to a testing set
    # 每个品种的图像,并将大致20%的图像划入测试集
    # 此函数返回的dog_breed即是image_filename_with_breed[0]也就是文件夹的名字即是狗的类别
    # breed_images则是一个迭代器是根据狗的类别进行分类的循环遍历dog_breed,输出breed_images则可以获得该品种下所有狗的图片元组
    # 但是groupby函数并不是将元组中狗的类别和狗的图片信息分开成两个数组，dog_breed中存储的是image_filename_with_breed[0],
    # breed_image是按狗的类别分类的image_filename_with_breed,所以breed_image[0]是狗的类别，breed_image[1]是狗图片的名称
    for i, breed_image in enumerate(breed_images):
        # breed_images表示当前狗的分类下的所有狗的图片，i表示当前遍历当前分类图片下的序号
        # 例如：表示哈巴狗分类下的第一张图片，哈巴狗分类下的第二张图片。
        if i%5 == 0:
            testing_dataset[dog_breed].append(breed_image[1])
        else:
            training_dataset[dog_breed].append(breed_image[1])
            #  表示其中五分之一加入测试集其余进入训练集
            #  并且以狗的类别名称进行区分,向同一类型中添加图片
            #  注意其中存储的并不是图片的信息，而是图片名称的信息。

    # 现在,每个字典就按照下列格式包含了所有的Chihuahua图像
    # training_dataset["n02085620-Chihuahua"] = ['./imagenet-dogs\\n02085620-Chihuahua\\n02085620_10131.jpg', ...]


def write_records_file(dataset, record_location):
    """
    Fill a TFRecords file with the images found in `dataset` and include their category.
    用dataset中的图像填充一个TFRecord文件,并将其类别包含进来
    Parameters
    参数
    ----------
    dataset : dict(list)
      Dictionary with each key being a label for the list of image filenames of its value.
      这个字典的键对应于其值中文件名列表对应的标签
    record_location : str
      Location to store the TFRecord output.
      存储TFRecord输出的路径
    """
    writer = None

    # Enumerating the dataset because the current index is used to breakup the files if they get over 100
    # images to avoid a slowdown in writing.
    # 枚举dataset,因为当前索引用于对文件进行划分,每个100幅图像,训练样本的信息就被写入到一个新的TFRecord文件中,以加快写操作的速度
    current_index = 0
    for breed, images_filenames in dataset.items():
        # print(breed)   n02085620-Chihuahua...
        # print(image_filenames)   ['./imagenet-dogs\\n02085620-Chihuahua\\n02085620_10074.jpg', ...]
        for image_filename in images_filenames:
            if current_index%100 == 0:  # 如果记录了100个文件的话,write就关闭
                if writer:
                    writer.close()
                # 否则开始记录write文件
                # record_Location表示当前的目录
                # current_index初始值为0,随着文件记录逐渐增加
                record_filename = "{record_location}-{current_index}.tfrecords".format(
                    record_location=record_location,
                    current_index=current_index)
                # format是格式化字符串操作,通过format(){}函数将文件名保存到record_filename中

                writer = tf.python_io.TFRecordWriter(record_filename)
            current_index += 1

            image_file = tf.read_file(image_filename)

            # In ImageNet dogs, there are a few images which TensorFlow doesn't recognize as JPEGs. This
            # try/catch will ignore those images.
            # 在ImageNet的狗的图像中,有少量无法被Tensorflow识别为JPEG的图像,利用try/catch可将这些图像忽略
            try:
                image = tf.image.decode_jpeg(image_file)
            except:
                print(image_filename)
                continue

            # Converting to grayscale saves processing and memory but isn't required.
            # 将其转化为灰度图片的类型,虽然这并不是必需的,但是可以减少计算量和内存占用,
            grayscale_image = tf.image.rgb_to_grayscale(image)
            resized_image = tf.image.resize_images(grayscale_image, (250, 151))  # 并将图片修改为长250宽151的图片类型

            # tf.cast is used here because the resized images are floats but haven't been converted into
            # image floats where an RGB value is between [0,1).
            # 这里之所以使用tf.cast,是因为 尺寸更改后的图像的数据类型是浮点数,但是RGB值尚未转换到[0,1)的区间之内
            image_bytes = sess.run(tf.cast(resized_image, tf.uint8)).tobytes()

            # Instead of using the label as a string, it'd be more efficient to turn it into either an
            # integer index or a one-hot encoded rank one tensor.
            # https://en.wikipedia.org/wiki/One-hot
            # 将标签按照字符串存储较为高效,推荐的做法是将其转换成整数索引或独热编码的秩1张量
            image_label = breed.encode("utf-8")

            example = tf.train.Example(features=tf.train.Features(feature={
                'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_label])),
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes]))
            }))

            writer.write(example.SerializeToString())  # 将其序列化为二进制字符串
    writer.close()


""" 使用write_records_file函数将testing_dataset和train_dataset 数据集写入tfrecords文件中"""
# 如果已经存在了TFrecords文件我们可以把此处注释掉
# write_records_file(testing_dataset, "./output/testing-images/testing-image")
# write_records_file(training_dataset, "./output/training-images/training-image")

# 读取Tfrecords文件
filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once("./output/training-images/*.tfrecords"))
#  生成文件名队列
reader = tf.TFRecordReader()
_, serialized = reader.read(filename_queue)
#  通过阅读器读取value值并将其保存为serialized

# 模板化的代码,将label和image分开
features = tf.parse_single_example(
    serialized,
    features={
        'label': tf.FixedLenFeature([], tf.string),
        'image': tf.FixedLenFeature([], tf.string),
    })

record_image = tf.decode_raw(features['image'], tf.uint8)
#  tf.decode_raw()函数将字符串的字节重新解释为一个数字的向量

# Changing the image into this shape helps train and visualize the output by converting it to
# be organized like an image.
# 修改图像的形状有助于训练和输出的可视化
image = tf.reshape(record_image, [250, 151, 1])

label = tf.cast(features['label'], tf.string)

min_after_dequeue = 10  # 当一次出列操作完成后,队列中元素的最小数量,往往用于定义元素的混合级别.
batch_size = 3  # 批处理大小
capacity = min_after_dequeue + 3*batch_size  # 批处理容量
image_batch, label_batch = tf.train.shuffle_batch(
    [image, label], batch_size=batch_size, capacity=capacity, min_after_dequeue=min_after_dequeue, num_threads=4)
# 通过随机打乱的方式创建数据批次
# 其中图片取出后会通过reshape方式添加一维数据通道
# 此处通过tf.train.shuffle_batch函数会添加一层batch_size数据。


# Converting the images to a float of [0,1) to match the expected input to convolution2d
# 将图像转换为灰度值位于[0, 1)的浮点类型,以与convlution2d期望的输入匹配
float_image_batch = tf.image.convert_image_dtype(image_batch, tf.float32)

#  第一个卷积层

conv2d_layer_one = tf.contrib.layers.conv2d(
    float_image_batch,
    num_outputs=32,  # 生成的滤波器的数量
    kernel_size=(5, 5),  # 滤波器的高度和宽度
    activation_fn=tf.nn.relu,
    weights_initializer=tf.random_normal_initializer,  # 设置weight的值是正态分布的随机值
    stride=(2, 2),  # 对image_batch和imput_channels的跨度值
    trainable=True)
# shape(3, 125, 76,32)
# 3表示批处理数据量是3,
# 125和76表示经过卷积操作后的宽和高,这和滤波器的大小还有步长有关系

#  第一个混合/池化层,输出降采样

pool_layer_one = tf.nn.max_pool(conv2d_layer_one,
                                ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1],
                                padding='SAME')
# shape(3, 63,38,32)
# 混合层ksize,1表示选取一个批处理数据,2表示在宽的维度取2个单位,2表示在高的取两个单位,1表示选取一个滤波器也就数选择一个通道进行操作.
# strides步长表示其分别在四个维度上的跨度

# Note, the first and last dimension of the convolution output hasn't changed but the
# middle two dimensions have.
# 注意卷积输出的第一个维度和最后一个维度没有发生变化,但是中间的两个维度发生了变化


# 第二个卷积层

conv2d_layer_two = tf.contrib.layers.conv2d(
    pool_layer_one,
    num_outputs=64,  # 更多输出通道意味着滤波器数量的增加
    kernel_size=(5, 5),
    activation_fn=tf.nn.relu,
    weights_initializer=tf.random_normal_initializer,
    stride=(1, 1),
    trainable=True)
# shape(3, 63,38,64)


# 第二个混合/池化层,输出降采样
pool_layer_two = tf.nn.max_pool(conv2d_layer_two,
                                ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1],
                                padding='SAME')
# shape(3, 32, 19,64)


# 光栅层，用于将所有数据光栅化，为全连接做准备
# 由于后面要使用softmax,因此全连接层需要修改为二阶张量,张量的第1维用于区分每幅图像,第二维用于对们每个输入张量的秩1张量
flattened_layer_two = tf.reshape(
    pool_layer_two,
    [
        batch_size,  # image_batch中的每幅图像
        -1  # 输入的其他所有维度
    ])
# 这里的-1参数将最后一个池化层调整为一个巨大的秩1张量


# 全连接层1
hidden_layer_three = tf.contrib.layers.fully_connected(
    flattened_layer_two,
    512,
    weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
    activation_fn=tf.nn.relu
)

# Dropout层
# 对一些神经元进行dropout操作.每个神经元以0.1的概率决定是否放电
hidden_layer_three = tf.nn.dropout(hidden_layer_three, 0.1)

# The output of this are all the connections between the previous layers and the 120 different dog breeds
# available to train on.
# 输出是前面的层与训练中可用的120个不同品种的狗的品种的全连接
# 全连接层2
final_fully_connected = tf.contrib.layers.fully_connected(
    hidden_layer_three,
    120,  # ImageNet Dogs 数据集中狗的品种数
    weights_initializer=tf.truncated_normal_initializer(stddev=0.1)
)

# 定义指标和训练
"""
由于每个标签都是字符串类型,tf.nn.softmax无法直接使用这些字符串,所以需要将这些字符创转换为独一无二的数字,
这些操作都应该在数据预处理阶段进行
"""
import glob

# Find every directory name in the imagenet-dogs directory (n02085620-Chihuahua, ...)
# 找到位于imagenet-dogs路径下的所有文件目录名
# glob操作用于查找定位系统内文件，支持通配符，split用于选取文件名部分中标签
labels = list(map(lambda c: c.split("/")[-1].split("\\")[1], glob.glob("./imagenet-dogs/*")))

# Match every label from label_batch and return the index where they exist in the list of classes
# 匹配每个来自label_batch的标签并返回它们在类别列表的索引
# 将label_batch作为参数l传入到匿名函数中tf.map_fn函数总体来讲和python中map函数相似,map_fn主要是将定义的函数运用到后面集合中每个元素中
train_labels = tf.map_fn(lambda l: tf.where(tf.equal(labels, l)
                                            )[0][0], label_batch, dtype=tf.int64)
# 关于这段代码，详见reference.md

# setup-only-ignore
loss = tf.reduce_mean(
    tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=final_fully_connected, labels=train_labels))

global_step = tf.Variable(0)  # 相当于global_step,是一个全局变量,在训练完一个批次后自动增加1

#  学习率使用退化学习率的方法
# 设置初始学习率为0.01,
learning_rate = tf.train.exponential_decay(learning_rate=0.01, global_step=global_step, decay_steps=120,
                                           decay_rate=0.95, staircase=True)

optimizer = tf.train.AdamOptimizer(learning_rate, 0.9).minimize(loss, global_step=global_step)

# 主程序
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
sess.run(init_op)

coord = tf.train.Coordinator()
# 线程控制管理器
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

# 训练
training_steps = 100
for step in range(training_steps):
    sess.run(optimizer)
    train_prediction = tf.nn.softmax(final_fully_connected)
    if step%10 == 0:
        print("loss:", sess.run(loss))

# print("prediction", sess.run(train_prediction))
# train_prediction = tf.nn.softmax(final_fully_connected)
# setup-only-ignore
filename_queue.close(cancel_pending_enqueues=True)
coord.request_stop()
coord.join(threads)
sess.close()
