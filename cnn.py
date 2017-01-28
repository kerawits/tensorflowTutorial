import tensorflow as tf
import math
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

MINIBATCH = 100
OUTPUT = 10

# Download images and labels into mnist.test (10K images+labels) and mnist.train (60K images+labels)
mnist = read_data_sets("data", one_hot=True, reshape=False, validation_size=0)

# training images
X = tf.placeholder(tf.float32, [None, 28, 28, 1])

# network
W1 = tf.Variable(tf.truncated_normal([6, 6, 1, 6], stddev=0.1), name='W1')
b1 = tf.Variable(tf.ones([6])/10, name='b1')

W2 = tf.Variable(tf.truncated_normal([5, 5, 6, 12], stddev=0.1), name='W2')
b2 = tf.Variable(tf.ones([12])/10, name='b2')

W3 = tf.Variable(tf.truncated_normal([4, 4, 12, 24], stddev=0.1), name='W3')
b3 = tf.Variable(tf.ones([24])/10, name='b3')

W4 = tf.Variable(tf.truncated_normal([7 * 7 * 24, 200], stddev=0.1), name='W4')
b4 = tf.Variable(tf.ones([200])/10, name='b4')

W5 = tf.Variable(tf.truncated_normal([200, OUTPUT], stddev=0.1), name='W5')
b5 = tf.Variable(tf.ones([OUTPUT])/10, name='b5')

# learning rate
learning_rate = tf.placeholder(tf.float32)

# dropout rate (1 for testing, 0.75 for training)
pkeep = tf.placeholder(tf.float32)

# model
Y1cnv = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')
Y1 = tf.nn.relu(Y1cnv + b1)
Y2cnv = tf.nn.conv2d(Y1, W2, strides=[1, 2, 2, 1], padding='SAME')
Y2 = tf.nn.relu(Y2cnv + b2)
Y3cnv = tf.nn.conv2d(Y2, W3, strides=[1, 2, 2, 1], padding='SAME')
Y3 = tf.nn.relu(Y3cnv + b3)
Y4 = tf.nn.relu(tf.matmul(tf.reshape(Y3, [-1, 7 * 7 * 24]), W4) + b4)
Y4d = tf.nn.dropout(Y4, pkeep)
Ylogits = tf.matmul(Y4d, W5) + b5
Y = tf.nn.softmax(Ylogits)

# labels
Y_ = tf.placeholder(tf.float32, [None, 10])

# loss function
# cross_entropy =  -tf.reduce_sum(Y_ * tf.log(Y))
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(Ylogits, Y_)
cross_entropy = tf.reduce_mean(cross_entropy) * 100

is_correct = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

optimizer = tf.train.AdamOptimizer(0.003)
train_step = optimizer.minimize(cross_entropy)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

MAXSTEP = 10000
lrmin = 0.0001
lrmax = 0.003
for i in range(MAXSTEP):
    batch_X, batch_Y = mnist.train.next_batch(MINIBATCH)
    lr = lrmin + (lrmax - lrmin) * math.exp(-i / MAXSTEP)

    train_data = {X: batch_X, Y_: batch_Y, learning_rate: lr, pkeep: 0.75}

    sess.run(train_step, feed_dict=train_data)

    if i % 10 == 0:
        train_data = {X: batch_X, Y_: batch_Y, learning_rate: lr, pkeep: 1.0}
        a, c = sess.run([accuracy, cross_entropy], feed_dict=train_data)
        print('step: {}: learning_rate: {} accuracy: {}, cross_entropy: {}'.format(i, lr, a, c))

    if i % 100 == 0:
        test_data = {X: mnist.test.images, Y_: mnist.test.labels, learning_rate: lr, pkeep: 1.0}
        a, c = sess.run([accuracy, cross_entropy], feed_dict=test_data)
        print('TEST: accuracy: {}, cross_entropy: {}'.format(a, c))

test_data = {X: mnist.test.images, Y_: mnist.test.labels, learning_rate: lr, pkeep: 1.0}
a, c = sess.run([accuracy, cross_entropy], feed_dict=test_data)
print('TEST: accuracy: {}, cross_entropy: {}'.format(a, c))
