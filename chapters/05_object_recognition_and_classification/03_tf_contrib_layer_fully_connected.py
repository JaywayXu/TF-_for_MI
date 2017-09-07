import tensorflow as tf

features = tf.constant([
    [[1.2], [3.4]]
])

fc = tf.contrib.layers.fully_connected(features, num_outputs=2)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(fc))
sess.close()
# [[[ 1.59672391  0.9847849 ]
#   [ 4.52405119  2.79022384]]]