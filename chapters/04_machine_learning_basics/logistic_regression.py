# Logistic regression example in TF using Kaggle's Titanic Dataset.
# Download train.csv from https://www.kaggle.com/c/titanic/data
# 该模型用于依据乘客的年龄,性别和船票的等级推断她或她是否可以幸存下来

import tensorflow as tf
import os

# same params and variables initialization as log reg.
# 与对数几率回归相同的参数和变量初始化
W = tf.Variable(tf.zeros([5, 1]), name="weights")
b = tf.Variable(tf.zeros([1]), name="bias")


# former inference is now used for combining inputs
# 之前的推断用于值的合并
def combine_inputs(X):
    return tf.matmul(X, W) + b


# new inferred value is the sigmoid applied to the former
# 新的推断值是将sigmoid函数运用到前面的合并值的输出
def inference(X):
    return tf.sigmoid(combine_inputs(X))


def loss(X, Y):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=combine_inputs(X), labels=Y))


# 编写读取文件的基本代码.
# 可以加载和解析 ,并创建一个批次来读取排列在张量中的多行数据
def read_csv(batch_size, file_name, record_defaults):
    filename_queue = tf.train.string_input_producer([os.path.join(os.getcwd(), file_name)])
    # os.getcwd()显示python当前工作路径
    # os.path.join连接目录和(文件名或目录)
    # tf.train.string_input_producer()获取文件名队列
    reader = tf.TextLineReader(skip_header_lines=1)
    # 跳过每个文件开头的skip_header_lines行
    key, value = reader.read(filename_queue)
    # 利用文件阅读器阅读文件名队列

    # decode_csv will convert a Tensor from type string (the text line) in
    # a tuple of tensor columns with the specified defaults, which also
    # sets the data type for each column
    """decode_csv可以将csv文件中的一行转化成一个tensor变量,
    具体的默认值以及默认的数据类型有record_defaults进行指定"""
    decoded = tf.decode_csv(value, record_defaults=record_defaults)

    # batch actually reads the file and loads "batch_size" rows in a single tensor
    return tf.train.shuffle_batch(decoded,
                                  batch_size=batch_size,
                                  capacity=batch_size*50,
                                  min_after_dequeue=batch_size)


def inputs():
    passenger_id, survived, pclass, name, sex, age, sibsp, parch, ticket, fare, cabin, embarked = \
        read_csv(100, "train.csv", [[0.0], [0.0], [0], [""], [""], [0.0], [0.0], [0.0], [""], [0.0], [""], [""]])

    # convert categorical data 分类数据转换.
    # 如果这里的pclass和[1]相等的话,返回[1.0],以此类推
    is_first_class = tf.to_float(tf.equal(pclass, [1]))
    is_second_class = tf.to_float(tf.equal(pclass, [2]))
    is_third_class = tf.to_float(tf.equal(pclass, [3]))

    gender = tf.to_float(tf.equal(sex, ["female"]))
    # 如果sex是female的话返回[1.0]

    # Finally we pack all the features in a single matrix;
    # We then transpose to have a matrix with one example per row and one feature per column.
    # 最后我们将所有特征排列在一个矩阵红,然后对矩阵转置,使其每行对应一个样本,每列对应一个特征.
    features = tf.transpose(tf.stack([is_first_class, is_second_class, is_third_class, gender, age]))
    # 在未打包前函数shape为(100,1),打包后为(5, 100,1),转置后shape为(1, 100, 5)
    survived = tf.reshape(survived, [100, 1])

    return features, survived


def train(total_loss):
    learning_rate = 0.01
    return tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)


def evaluate(sess, X, Y):
    predicted = tf.cast(inference(X) > 0.5, tf.float32)
    # print("the perdiction is ", predicted)
    # the perdiction is  Tensor("Cast:0", shape=(100, 1), dtype=float32)

    print(sess.run(tf.reduce_mean(tf.cast(tf.equal(predicted, Y), tf.float32))))


# Launch the graph in a session, setup boilerplate
with tf.Session() as sess:
    tf.global_variables_initializer().run()

    X, Y = inputs()

    total_loss = loss(X, Y)
    train_op = train(total_loss)  # 训练函数

    coord = tf.train.Coordinator()  # 线程管理器
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)  # 开始线程

    # actual training loop
    training_steps = 1000
    for step in range(training_steps):
        sess.run(train_op)

        if step%10 == 0:
            print("loss: ", sess.run(total_loss))

    evaluate(sess, X, Y)

    import time

    time.sleep(5)

    coord.request_stop()
    coord.join(threads)
    sess.close()
"""equal函数支持广播模式
import tensorflow as tf

a = tf.constant([[1], [2], [3], [1]])  shape为(4, 1)
b = tf.to_float(tf.equal(a, [1]))  shape为(1)
with tf.Session() as sess:
    print(sess.run(b))
# [[ 1.]
#  [ 0.]
#  [ 0.]
#  [ 1.]]"""
"""cast函数与条件语句混用
import tensorflow as tf

a = tf.constant([[0.3], [0.5], [0.6], [0.7]])
predicted = tf.cast(a > 0.5, tf.float32)
with tf.Session() as sess:
    print(sess.run(predicted))
# [[ 0.]
#  [ 0.]
#  [ 1.]
#  [ 1.]]
"""