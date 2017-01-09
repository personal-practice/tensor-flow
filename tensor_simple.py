import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# inputs
x = tf.placeholder(tf.float32, [None, 784])

# weights
W = tf.Variable(tf.zeros([784, 10])) 

# biases
b = tf.Variable(tf.zeros([10])) 

# model (softmax)
y = tf.nn.softmax(tf.matmul(x, W) + b)

# correct answers
y_ = tf.placeholder(tf.float32, [None, 10]) # 

# cost function (cross entropy)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# optimization algorithm (gradient descent)
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# initialize variables
init = tf.initialize_all_variables()

# execute
sess = tf.Session()
sess.run(init)

# train (using stochastic gradient descent)
for i in range(1000):
	batch_xs, batch_ys = mnist.train.next_batch(100)
	sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_:mnist.test.labels}))