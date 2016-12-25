import tensorflow as tf
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.curdir)[:-8])

import math_ops

# Starting interactive Session
sess = tf.InteractiveSession()

true_tensor = tf.constant([[20,30,40,50,60],[2,3,4,5,0]])
false_tensor = tf.constant([[2,3,4,5,0],[20,30,40,50,60]])
cond = tf.constant([True, False])

cond_gathered = math_ops.cond_gather(cond, true_tensor, false_tensor)

sess.run(tf.initialize_all_variables())

print(true_tensor.eval().shape)
print(cond_gathered.eval())

sess.close()
