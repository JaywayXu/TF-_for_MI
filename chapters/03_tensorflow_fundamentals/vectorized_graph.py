import tensorflow as tf
import numpy as np

# 新建一个图的对象,并将其设置为默认图
# 显式创建一个图加以使用,而非使用默认的图
graph = tf.Graph()

with graph.as_default():
    # 在图中有两个"全局风格"的Variable对象由于这些对象在本质上是全局的,因此在声明时与数据流图中其他节点区分开,将他们放入自己的名称作用域
    with tf.name_scope("variables"):
        # 记录数据流程图运行次数的Variable对象
        global_step = tf.Variable(0, dtype=tf.int32, name="global_step")

        # 追踪该模型的所有输出随时间的累加和的Variable对象
        total_output = tf.Variable(0.0, dtype=tf.float32, name="total_output")

    # 核心变换操作
    with tf.name_scope("transformation"):
        # 独立的输入层
        with tf.name_scope("input"):
            # 创建输出占位符,用于接收一个向量
            a = tf.placeholder(tf.float32, shape=[None], name="input_placeholder_a")

        # 独立的中间层
        with tf.name_scope("intermediate_layer"):
            b = tf.reduce_prod(a, name="product_b")  # 沿着指定维度计算元素乘积
            c = tf.reduce_sum(a, name="sum_c")

        # 独立的输出层
        with tf.name_scope("output"):
            output = tf.add(b, c, name="output")

    with tf.name_scope("update"):
        # 用最新的输出更新Variable对象total_output
        update_total = total_output.assign_add(output)

        # 将前面的Variable对象global_step增1,只要数据流图运行,该操作便需要运行.
        increment_step = global_step.assign_add(1)

    # 总结操作
    with tf.name_scope("summaries"):
        avg = tf.div(update_total, tf.cast(increment_step, tf.float32), name="average")
        # cast函数用于将int型数据转换为tf.float32数据类型
        # 计算随时间输出的均值,获取当前全部输出的总和total_output(使用来自update_total的输出,以确保在计算avg之前更新便已经全部完成)
        # 以及数据流图的总运行次数global_step(使用increment_step的输出,以确保数据流图有序运行)
        # 为输出结点创建汇总数据
        tf.summary.scalar(name="output_summary", tensor=output)
        tf.summary.scalar(name="total_summary", tensor=update_total)
        tf.summary.scalar(name="average_summary", tensor=avg)

    # 全局变量和操作
    # 为完成数据流图的构建,还需要创建Variable对象初始化Op和用于将所有汇总数据组织到一个Op的辅助结点,把他们放进名为"global_ops"的名称作用域
    with tf.name_scope("global_ops"):
        # 初始化所有的变量
        init = tf.initialize_all_variables()
        # 合并所有的汇总数据组织到一个Op的辅助节点
        merged_summaries = tf.summary.merge_all()
        """将merge_all_summaries()与其他全局OPs放在一起是最佳做法,这可以想象为一个拥有Variable对象,Op和名称作用域等的不同汇总数据的数据流图"""

# 使用显式创建的图形开始一个会话
sess = tf.Session(graph=graph)

# 用于保存汇总数据
writer = tf.summary.FileWriter('./improved_graph', graph)

# 初始化所有变量
sess.run(init)


def run_graph(input_tensor):
    """
    帮助函数; 利用给定的张量作为输入并且保存汇总数据
    """
    feed_dict = {a: input_tensor}
    # 其中a是一个占位符,用于输入数据,input_tensor表示a变量的数据e
    out, step, summary = sess.run([output, increment_step, merged_summaries], feed_dict=feed_dict)

    # sesstion.run可以运行依次列表中[output,increment_step,merged_summaries]
    # 其中output表示程序运行的结果,increment_step表示程序的step步数,merged_summaries表示各种总结数据

    writer.add_summary(summary, global_step=step)
    # global_step参数十分重要,因为他是Tensorflow可以随着时间对数据进行图示.


# Run the graph with various inputs
run_graph([2, 8])
run_graph([3, 1, 3, 3])
run_graph([8])
run_graph([1, 2, 3])
run_graph([11, 4])
run_graph([4, 1])
run_graph([7, 3, 1])
run_graph([6, 3])
run_graph([0, 2])
run_graph([4, 5, 6])

# 将汇总数据写入磁盘
writer.flush()

# 关闭writer
writer.close()

# Close the session
sess.close()

# To start TensorBoard after running this file, execute the following command:
# $ tensorboard --logdir=F://Git/TF-_for_MI/chapters/03_tensorflow_fundamentals/improved_graph
