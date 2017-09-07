"""tf.nn.avg_pool跳跃遍历一个张量,并将被卷积核覆盖的各深度值取平均.
当整个卷积核都非常重要时,若虚实现值的所见,平均池化非常有用
例如输入张量宽度和高度都很大但是深度很小的情况"""
import tensorflow as tf

batch_size = 1
input_height = 3
input_width = 3
input_channels = 1

layer_input = tf.constant([
    [
        [[1.0], [1.0], [1.0]],
        [[1.0], [0.5], [0.0]],
        [[0.0], [0.0], [0.0]]
    ]
])
sess = tf.Session()

# The strides will look at the entire input by using the image_height and image_width
# strides会使用image_height和image_width遍历整个输入
kernel = [batch_size, input_height, input_width, input_channels]
max_pool = tf.nn.avg_pool(layer_input, kernel, [1, 1, 1, 1], "VALID")
print(sess.run(tf.shape(layer_input)))
# [1 3 3 1]
print(sess.run(max_pool))
# [[[[ 0.5]]]]
# 这便达到了改变输出的尺寸的目的.