import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

MINIBATCH = 100

# Download images and labels into mnist.test (10K images+labels) and mnist.train (60K images+labels)
mnist = read_data_sets("data", one_hot=True, reshape=False, validation_size=0)

# training images
X = tf.placeholder(tf.float32, [None, 28, 28, 1])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# model
Y = tf.nn.softmax(tf.matmul(tf.reshape(X, [-1, 784]), W) + b)

# labels
Y_ = tf.placeholder(tf.float32, [None, 10])

# loss function
cross_entropy =  -tf.reduce_sum(Y_ * tf.log(Y))

is_correct = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

optimizer = tf.train.GradientDescentOptimizer(0.003)
train_step = optimizer.minimize(cross_entropy)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for i in range(2000):
    batch_X, batch_Y = mnist.train.next_batch(MINIBATCH)
    train_data = {X: batch_X, Y_: batch_Y}

    sess.run(train_step, feed_dict=train_data)

    if i%10 == 0:
        a, c = sess.run([accuracy, cross_entropy], feed_dict=train_data)
        print('step: {}: accuracy: {}, cross_entropy: {}'.format(i, a, c))

    if i%100 == 0:
        test_data = {X: mnist.test.images, Y_: mnist.test.labels}
        a, c = sess.run([accuracy, cross_entropy], feed_dict=test_data)
        print('TEST: accuracy: {}, cross_entropy: {}'.format(a, c))

test_data = {X: mnist.test.images, Y_: mnist.test.labels}
a, c = sess.run([accuracy, cross_entropy], feed_dict=test_data)
print('TEST: accuracy: {}, cross_entropy: {}'.format(a, c/100))
