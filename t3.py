import tensorflow as tf 
import numpy as np 
a = tf.placeholder("float")
b = tf.add(a,3.0)
sess =tf.Session()
print(sess.run(b,feed_dict={a:2.0}))
sess.close()