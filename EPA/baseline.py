import tensorflow as tf
from problem import Data

if __name__ == '__main__':
    n = 100
    niter = 100000
    inputsize = 1
    outputsize = 1
    bs = 10
    lr = 0.01

    # create a silly task to train on: a step function
    data = Data(f=lambda x: [1] if x>n/2 else [0], n=n)
    data.shuffle()
    
    # nn baseline
    sess = tf.Session()    
    x = tf.placeholder(tf.float32, shape=[None, inputsize])
    y_ = tf.placeholder(tf.float32, shape=[None, outputsize])

    W = tf.Variable(tf.zeros([inputsize,outputsize]))
    b = tf.Variable(tf.zeros([outputsize]))
    sess.run(tf.initialize_all_variables())

    y = tf.nn.sigmoid(tf.matmul(x,W) + b)
    se_loss = tf.reduce_mean(tf.square(y-y_)) # squared error
    train_step = tf.train.GradientDescentOptimizer(lr).minimize(se_loss)

    for i in range(niter):
        batch = data.getTr().getBatch(bs)
        train_step.run(feed_dict={x: batch.x.reshape(bs,1), y_: batch.y}, session=sess)
        # print the total progress
        print(se_loss.eval(feed_dict={x: data.getTe().x[...,0], y_: data.getTe().y}, session=sess))

    # cnn baseline: TODO
    
    # rnn baseline: TODO

    # conv rnn baseline: TODO

    
