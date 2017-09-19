"""转换图像数据格式时需要将它们的颜色空间变为灰度空间,将图像尺寸修改为同一尺寸,并将标签依附于每幅图像"""
import tensorflow as tf

sess = tf.InteractiveSession()
import glob

image_filenames = glob.glob("./imagenet-dogs/n02*/*.jpg")  # 访问imagenet-dogs文件夹中所有n02开头的子文件夹中所有的jpg文件

# image_filenames[0:2]  此语句表示image_filenames文件中的从第0个编号到第2个编号的值
# ['./imagenet-dogs\\n02085620-Chihuahua\\n02085620_10074.jpg',
# './imagenet-dogs\\n02085620-Chihuahua\\n02085620_10131.jpg']
# 此时image_filenames中保存的全部是类似于以上形式的值
# 注意书上的解释和这个输出和此处的输出与有很大的不同,原因是书本是用linux系统,
# 所以是以"/"对文件名进行分隔符的操作而此处不是windows下使用"\\"对文件名进行操作.

from itertools import groupby
from collections import defaultdict

training_dataset = defaultdict(list)
testing_dataset = defaultdict(list)

# Split up the filename into its breed and corresponding filename. The breed is found by taking the directory name
# 将文件名分解为品种和对应的文件名,品种对应于文件夹名称
image_filename_with_breed = map(lambda filename: (filename.split("/")[1].split("\\")[1], filename), image_filenames)
# 表示定义一个匿名函数lambda传入参数为filename,对filename以"/"为分隔符,然后取第二个值,并且返回filename.split("/")[1]和filename
# 并且以image_filenames作为参数
# ('n02086646-Blenheim_spaniel', './imagenet-dogs\\n02086646-Blenheim_spaniel\\n02086646_3739.jpg')

# Group each image by the breed which is the 0th element in the tuple returned above
# 依据品种(上述返回的元组的第0个分量对元素进行分组)
for dog_breed, breed_images in groupby(image_filename_with_breed, lambda x: x[0]):
    # Enumerate each breed's image and send ~20% of the images to a testing set
    # 美剧每个品种的图像,并将大致20%的图像划入测试集
    # 此函数返回的dog_breed即是image_filename_with_breed[0]也就是文件夹的名字即是狗的类别
    # breed_images则是一个迭代器是根据狗的类别进行分类的
    for i, breed_image in enumerate(breed_images):
        #  breed_images此时是根据狗的种类进行分类的迭代器
        #  返回的i表示品种的代表编号
        #  返回的breed_image表示这个标号的种类下狗的图片
        if i%5 == 0:
            testing_dataset[dog_breed].append(breed_image[1])
        else:
            training_dataset[dog_breed].append(breed_image[1])
        #  表示其中五分之一加入测试集其余进入训练集
        #  并且以狗的类别名称进行区分,向同一类型中添加图片
    # Check that each breed includes at least 18% of the images for testing
    breed_training_count = len(training_dataset[dog_breed])
    breed_testing_count = len(testing_dataset[dog_breed])
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
            try:
                image = tf.image.decode_jpeg(image_file)
            except:
                print(image_filename)
                continue

            # Converting to grayscale saves processing and memory but isn't required.
            grayscale_image = tf.image.rgb_to_grayscale(image)
            resized_image = tf.image.resize_images(grayscale_image, [250, 151])

            # tf.cast is used here because the resized images are floats but haven't been converted into
            # image floats where an RGB value is between [0,1).
            image_bytes = sess.run(tf.cast(resized_image, tf.uint8)).tobytes()

            # Instead of using the label as a string, it'd be more efficient to turn it into either an
            # integer index or a one-hot encoded rank one tensor.
            # https://en.wikipedia.org/wiki/One-hot
            image_label = breed.encode("utf-8")

            example = tf.train.Example(features=tf.train.Features(feature={
                'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_label])),
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes]))
            }))

            writer.write(example.SerializeToString())
    writer.close()


write_records_file(testing_dataset, "./output/testing-images/testing-image")
write_records_file(training_dataset, "./output/training-images/training-image")
filename_queue = tf.train.string_input_producer(
    tf.train.match_filenames_once("./output/training-images/*.tfrecords"))
reader = tf.TFRecordReader()
_, serialized = reader.read(filename_queue)

features = tf.parse_single_example(
    serialized,
    features={
        'label': tf.FixedLenFeature([], tf.string),
        'image': tf.FixedLenFeature([], tf.string),
    })

record_image = tf.decode_raw(features['image'], tf.uint8)

# Changing the image into this shape helps train and visualize the output by converting it to
# be organized like an image.
image = tf.reshape(record_image, [250, 151, 1])

label = tf.cast(features['label'], tf.string)

min_after_dequeue = 10
batch_size = 3
capacity = min_after_dequeue + 3*batch_size
image_batch, label_batch = tf.train.shuffle_batch(
    [image, label], batch_size=batch_size, capacity=capacity, min_after_dequeue=min_after_dequeue)
# Converting the images to a float of [0,1) to match the expected input to convolution2d
float_image_batch = tf.image.convert_image_dtype(image_batch, tf.float32)

conv2d_layer_one = tf.contrib.layers.conv2d(
    float_image_batch,
    num_outputs=32,  # The number of filters to generate
    kernel_size=(5, 5),  # It's only the filter height and width.
    activation_fn=tf.nn.relu,
    weights_initializer=tf.random_normal_initializer,
    stride=(2, 2),
    trainable=True)

pool_layer_one = tf.nn.max_pool(conv2d_layer_one,
                                ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1],
                                padding='SAME')

# Note, the first and last dimension of the convolution output hasn't changed but the
# middle two dimensions have.
conv2d_layer_one.get_shape(), pool_layer_one.get_shape()

conv2d_layer_two = tf.contrib.layers.conv2d(
    pool_layer_one,
    num_outputs=64,  # More output channels means an increase in the number of filters
    kernel_size=(5, 5),
    activation_fn=tf.nn.relu,
    weights_initializer=tf.random_normal_initializer,
    stride=(1, 1),
    trainable=True)

pool_layer_two = tf.nn.max_pool(conv2d_layer_two,
                                ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1],
                                padding='SAME')

conv2d_layer_two.get_shape(), pool_layer_two.get_shape()
flattened_layer_two = tf.reshape(
    pool_layer_two,
    [
        batch_size,  # Each image in the image_batch
        -1  # Every other dimension of the input
    ])

flattened_layer_two.get_shape()
# The weight_init parameter can also accept a callable, a lambda is used here  returning a truncated normal
# with a stddev specified.
hidden_layer_three = tf.contrib.layers.fully_connected(
    flattened_layer_two,
    512,
    weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
    activation_fn=tf.nn.relu
)

# Dropout some of the neurons, reducing their importance in the model
hidden_layer_three = tf.nn.dropout(hidden_layer_three, 0.1)

# The output of this are all the connections between the previous layers and the 120 different dog breeds
# available to train on.
final_fully_connected = tf.contrib.layers.fully_connected(
    hidden_layer_three,
    120,  # Number of dog breeds in the ImageNet Dogs dataset
    weights_initializer=tf.truncated_normal_initializer(stddev=0.1)
)
import glob

# Find every directory name in the imagenet-dogs directory (n02085620-Chihuahua, ...)
labels = list(map(lambda c: c.split("/")[-1], glob.glob("./imagenet-dogs/*")))

# Match every label from label_batch and return the index where they exist in the list of classes
train_labels = tf.map_fn(lambda l: tf.where(tf.equal(labels, l))[0, 0:1][0], label_batch, dtype=tf.int64)
# setup-only-ignore
loss = tf.reduce_mean(
    tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=final_fully_connected, labels=train_labels))

batch = tf.Variable(0)
learning_rate = tf.train.exponential_decay(
    0.01,
    batch*3,
    120,
    0.95,
    staircase=True)

optimizer = tf.train.AdamOptimizer(
    learning_rate, 0.9).minimize(
    loss, global_step=batch)

train_prediction = tf.nn.softmax(final_fully_connected)
# setup-only-ignore
filename_queue.close(cancel_pending_enqueues=True)
coord.request_stop()
coord.join(threads)
