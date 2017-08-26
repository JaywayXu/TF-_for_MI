# TF code scaffolding for building simple models.
# 为模型训练和评估定义一个通用的代码框架
import tensorflow as tf


# 初始化变量和模型参数,定义训练闭环中的运算
# initialize variables/model parameters
# define the training loop operations
def inference(X):
    # compute inference model over data X and return the result
    # 计算推断模型在数据X上的输出,并将结果返回
    return


def loss(X, Y):
    # compute loss over training data X and expected values Y
    # 依据训练数据X及其期望输出Y计算损失
    return


def inputs():
    # read/generate input training data X and expected outputs Y
    # 读取或生成训练数据X及其期望输出Y
    return


def train(total_loss):
    # train / adjust model parameters according to computed total loss
    # 依据计算的总损失训练或调整模型参数
    return


def evaluate(sess, X, Y):
    # evaluate the resulting trained model
    # 对训练得到的模型进行评估
    return


# Launch the graph in a session, setup boilerplate
# 在一个回话对象红启动数据流图,搭建流程
with tf.Session() as sess:
    tf.initialize_all_variables().run()

    X, Y = inputs()

    total_loss = loss(X, Y)
    train_op = train(total_loss)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # actual training loop
    # 实际的训练闭环迭代次数
    training_steps = 1000
    for step in range(training_steps):
        sess.run([train_op])
        # for debugging and learning purposes, see how the loss gets decremented thru training steps
        # 处于调试和学习的目的,查看损失在训练从过程中递减的情况
        if step%10 == 0:
            print("loss: ", sess.run([total_loss]))

    evaluate(sess, X, Y)

    coord.request_stop()
    coord.join(threads)
    sess.close()
