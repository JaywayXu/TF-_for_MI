# Softmax example in TF using the classical Iris dataset
# Download iris.data from https://archive.ics.uci.edu/ml/datasets/Iris

import tensorflow as tf
import os

# this time weights form a matrix, not a column vector, one "weight vector" per class.
# 此时,权值构成了一个矩阵,而非向量,每个"特征权值列"对应一个输出类别
# 该数据及包括了4个数据特横以及三个可能的输出类所以权值矩阵的维数应为4*3
W = tf.Variable(tf.zeros([4, 3]), name="weights")
# so do the biases, one per class.
# 偏置也是如此,每一个偏置对应一个输出类
b = tf.Variable(tf.zeros([3], name="bias"))


def combine_inputs(X):
    return tf.matmul(X, W) + b
    # 这里使用的高阶张量相乘的情况


def inference(X):
    return tf.nn.softmax(combine_inputs(X))
# 输入是一个shape为(100,3)的张量,根据softmax函数的定义,返回的函数和输入的logits值有一样的shape也为(100,3)


def loss(X, Y):
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=combine_inputs(X), labels=Y))


# 此函数对于每个样本只对应单个类别进行了专门的优化


def read_csv(batch_size, file_name, record_defaults):
    filename_queue = tf.train.string_input_producer([os.path.dirname(__file__) + "/" + file_name])

    reader = tf.TextLineReader(skip_header_lines=1)
    key, value = reader.read(filename_queue)  # 读取文件名列表返回键值对

    # decode_csv will convert a Tensor from type string (the text line) in
    # a tuple of tensor columns with the specified defaults, which also
    # sets the data type for each column
    # 将csv文件中的一行转化为tensor形式.record_defaults指定默认的填充值,和读取列时列的数据类型.
    decoded = tf.decode_csv(value, record_defaults=record_defaults)

    # batch actually reads the file and loads "batch_size" rows in a single tensor
    # 批处理文件函数实际读取文件,并且读取batch_size行进一个单独的tensor中
    return tf.train.shuffle_batch(decoded,
                                  batch_size=batch_size,
                                  capacity=batch_size*50,
                                  num_threads=1,
                                  min_after_dequeue=batch_size)


def inputs():
    sepal_length, sepal_width, petal_length, petal_width, label = \
        read_csv(100, "iris.data", [[0.0], [0.0], [0.0], [0.0], [""]])
    # 数据集中有四格float类型的属性特征,最后一个是label.

    # convert class names to a 0 based class index.
    # 将类名称转换为从零开始计的类别索引
    # 此时label.shape=(100,1)表示100个样本的类别此时tf.argmax函数处理的是一个shape为(3,100,1)的tensor
    # 此时返回第0阶上的最大值,则此时的shape为(100,1)其中储存的是从0开始的类别的下标
    label_number = tf.to_int32(tf.argmax(tf.to_int32(tf.stack([
        tf.equal(label, ["Iris-setosa"]),
        tf.equal(label, ["Iris-versicolor"]),
        tf.equal(label, ["Iris-virginica"])
    ])), 0))

    # Pack all the features that we care about in a single matrix;
    # We then transpose to have a matrix with one example per row and one feature per column.
    # 将所关心的所有特征装入单个矩阵中,然后对该矩阵转置,使其每行对应一个样本,每列对应一个特征.
    # 这样就要使特征值满足shape为(100,4)的情况经过转置操作后shape为(1,100,4)
    """利用stack函数创建一个张量,并利用tf.equal()函数将文件输入与每个可能的值进行比较,
    然后,利用tf.argmax找到张量中值为真的位置,从而将各类别转化为0~2范围内的整数"""
    features = tf.transpose(tf.stack([sepal_length, sepal_width, petal_length, petal_width]))

    return features, label_number


def train(total_loss):
    # 对于total_loss,这里实际计算的是loss函数计算得到的损失函数
    learning_rate = 0.01
    return tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)
    # 使用随机梯度下降方法对total_loss进行优化.


def evaluate(sess, X, Y):
    predicted = tf.cast(tf.arg_max(inference(X), 1), tf.int32)
    # inference(X).shape=(100,3)这里利用argmax函数在第一维进行降维操作,返回的是第一维上的最大值的索引perdicted.shape=(100)

    print(sess.run(tf.reduce_mean(tf.cast(tf.equal(predicted, Y), tf.float32))))
    # 如果和标签相同的话,返回1,计算平均值.

# Launch the graph in a session, setup boilerplate
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()

    X, Y = inputs()

    total_loss = loss(X, Y)
    train_op = train(total_loss)

    coord = tf.train.Coordinator()
    # 线程控制管理器
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # actual training loop
    training_steps = 1000
    for step in range(training_steps):
        sess.run(train_op)
        # for debugging and learning purposes, see how the loss gets decremented thru training steps
        if step%10 == 0:
            print("loss: ", sess.run(total_loss))

    evaluate(sess, X, Y)
    coord.request_stop()
    coord.join(threads)
    sess.close()
