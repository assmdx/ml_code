import tensorflow as tf
import sys
print(sys.maxunicode)
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))