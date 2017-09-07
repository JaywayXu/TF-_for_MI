"""池化层能够减少拟合,并通过减小输入的尺寸来提高性能,可用于对输入降采样,但会为后续层保留重要的信息.
tf.nn.max_pool
跳跃遍历某个张量,并从被卷积核覆盖的元素中找到最大的数值作为卷积结果,
当输入的数据灰度与图像中的重要性相关时,这种池化方式非常有用
"""
# Usually the input would be output from a previous layer and not an image directly.
# 输入通常为前一层的输出而非直接为图像.
import tensorflow as tf

sess = tf.Session()
batch_size = 1
input_height = 3
input_width = 3
input_channels = 1

layer_input = tf.constant([
    [
        [[1.0], [0.2], [1.5]],
        [[0.1], [1.2], [1.4]],
        [[1.1], [0.4], [0.4]]
    ]
])

# The strides will look at the entire input by using the image_height and image_width
# strides会使用image_height和image_width遍历整个输入.
kernel = [batch_size, input_height, input_width, input_channels]
max_pool = tf.nn.max_pool(layer_input, kernel, [1, 1, 1, 1], "VALID")
# print(sess.run(tf.shape(layer_input)))
# [1 3 3 1]
print(sess.run(max_pool))
sess.close()
# [[[[ 1.5]]]]
"""layer_input是一个形状类似于tf.nn.conv2d或某个激活函数的输出的张量.
目标是仅保留一个值,即该张量中的最大元素,在本例中,该张量的最大分量是1.5,并以输入相同的格式被返回.
最大池化(max-pooling)通常是利用2*2的接受域(高度和宽度均为2的卷积核)完成的,通常被称为"2*2的最大池化运算"
使用2*2的接受域的原因之一在于他是在单个通路上能够实施的最小数量的降采样.如果使用1*1的接受域,则输出将与输入相同.
"""