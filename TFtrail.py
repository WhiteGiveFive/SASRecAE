import tensorflow as tf
import numpy as np
import time

value = np.random.randn(5000, 1000)
a = tf.constant(value)

b = a * a

c = 0
tic = time.time()
with tf.Session() as sess:
    for i in range(1000):
        sess.run(b)

        c += 1
        if c % 100 == 0:
            d = c / 10
            # print(d)
            print("Computation proceeds%s%%" % d)

toc = time.time()
t_cost = toc - tic

print("Time spent in testing%s" % t_cost)
print("Ubuntu upper GPU 1050 ti Test time is 7.99727988243103")


state = tf.Variable(0, name='counter')
one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    sess.run(update)
    print(sess.run(state))
    print(update)
