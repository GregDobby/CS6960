# encoder and decoder for logistic function gradient
import tensorflow as tf
import numpy as np
import math

def main():
    # encoder and decoder configuration
    dim = 5  # gradient dimension
    npt = 10 # number of points in each worker node
    nw = 5 # number of worker
    nr = 1 # redundancy, number of encoder outputs
    nu = 1 # number of unreliable sources
    hn = 50 
    enc_layer_size = [dim * nw * npt, hn, hn, hn, dim * npt] # encoder layer sizes
    dec_layer_size = [dim * (nw + nr), hn, hn, hn, dim * nw] # decoder layer sizes

    # encoder input
    x_enc = tf.placeholder(tf.float32, [None, enc_layer_size[0]])

    ## encoder
    # layer 1
    ew1 = tf.Variable(tf.random_normal([enc_layer_size[0], enc_layer_size[1]], stddev=math.sqrt(2 / (enc_layer_size[0] + enc_layer_size[1]))))
    eb1 = tf.Variable(tf.zeros(enc_layer_size[1]))
    ey1 = tf.nn.tanh(tf.matmul(x_enc, ew1) + eb1)

    # layer 2
    ew2 = tf.Variable(tf.random_normal([enc_layer_size[1], enc_layer_size[2]], stddev=math.sqrt(2 / (enc_layer_size[1] + enc_layer_size[2]))))
    eb2 = tf.Variable(tf.zeros(enc_layer_size[2]))
    ey2 = tf.nn.tanh(tf.matmul(ey1, ew2) + eb2)

    # layer 3
    ew3 = tf.Variable(tf.random_normal([enc_layer_size[2], enc_layer_size[3]], stddev=math.sqrt(2 / (enc_layer_size[2] + enc_layer_size[3]))))
    eb3 = tf.Variable(tf.zeros(enc_layer_size[3]))
    ey3 = tf.nn.tanh(tf.matmul(ey2, ew3) + eb3)

    # layer 4
    ew4 = tf.Variable(tf.random_normal([enc_layer_size[3], enc_layer_size[4]], stddev=math.sqrt(2 / (enc_layer_size[3] + enc_layer_size[4]))))
    eb4 = tf.Variable(tf.zeros(enc_layer_size[4]))
    x_coded = tf.matmul(ey3, ew4) + eb4 # encoder output

    # decoder input
    g = tf.placeholder(tf.float32, [None, dim * nw])
    w = tf.placeholder(tf.float32, [None, dim])
    x_coded = tf.reshape(x_coded,[-1, dim])
    w_reshaped = tf.reshape(tf.tile(w, [1, npt]), [-1, dim])
    wx = tf.reduce_sum(tf.multiply(x_coded, -w_reshaped), 1)
    f = tf.reshape(tf.divide(1 , 1 +  tf.exp(wx)),[-1,1])
    new_g = tf.multiply(tf.tile(tf.multiply(f, 1 - f), [1,dim]), x_coded)
    new_g = tf.reduce_sum(tf.reshape(new_g, [npt, -1]),0)
    new_g = tf.reshape(new_g,[-1, dim])


    mask = tf.placeholder(tf.float32, [None, dec_layer_size[0]]) # unreliable pattern
    g_dec = tf.multiply(tf.concat([g, new_g], 1), mask)
    
    ## decoder
    # layer 1
    dw1 = tf.Variable(tf.random_normal([dec_layer_size[0], dec_layer_size[1]], stddev=math.sqrt(2 / (dec_layer_size[0] + dec_layer_size[1]))))
    db1 = tf.Variable(tf.zeros(dec_layer_size[1]))
    dy1 = tf.nn.tanh(tf.matmul(g_dec, dw1) + db1)

    
    # layer 2
    dw2 = tf.Variable(tf.random_normal([dec_layer_size[1], dec_layer_size[2]], stddev=math.sqrt(2 / (dec_layer_size[1] + dec_layer_size[2]))))
    db2 = tf.Variable(tf.zeros(dec_layer_size[2]))
    dy2 = tf.nn.tanh(tf.matmul(dy1, dw2) + db2)

    # layer 3 
    dw3 = tf.Variable(tf.random_normal([dec_layer_size[2], dec_layer_size[3]], stddev=math.sqrt(2 / (dec_layer_size[2] + dec_layer_size[3]))))
    db3 = tf.Variable(tf.zeros(dec_layer_size[3]))
    dy3 = tf.nn.tanh(tf.matmul(dy2, dw3) + db3)

    # layer 4
    dw4 = tf.Variable(tf.random_normal([dec_layer_size[3], dec_layer_size[4]], stddev=math.sqrt(2 / (dec_layer_size[3] + dec_layer_size[4]))))
    db4 = tf.Variable(tf.zeros(dec_layer_size[4]))
    g_decoded = tf.matmul(dy3, dw4) + db4

    # loss
    loss = tf.reduce_mean(tf.sqrt(tf.reduce_mean(((g - g_decoded)/g)**2, axis = 1)))

    # optimizer
    lr = 0.01   # learning rate
    opt = tf.train.AdamOptimizer(lr)
    train = opt.minimize(loss)

    # init
    init_op = tf.global_variables_initializer()

    total_step = 5050505050000
    with tf.Session() as sess:
        sess.run(init_op)
        ## load data
        # train
        train_w = np.genfromtxt('./train_w.csv', delimiter=',')
        train_x = np.genfromtxt('./train_x.csv', delimiter=',')
        train_g = np.genfromtxt('./train_g.csv', delimiter=',')
        train_mask = np.genfromtxt('./train_mask.csv', delimiter=',')
        # test
        test_w = np.genfromtxt('./test_w.csv', delimiter=',')
        test_x = np.genfromtxt('./test_x.csv', delimiter=',')
        test_g = np.genfromtxt('./test_g.csv', delimiter=',')
        test_mask = np.genfromtxt('./test_mask.csv', delimiter=',')

        feed_dict = {x_enc: train_x, mask: train_mask, w: train_w, g: train_g}
        for i in range(total_step):
            _,train_err =  sess.run([train, loss], feed_dict=feed_dict)
            if i % 100 == 0:
                print('train_error:', train_err)

        # test
        feed_dict = {x_enc: test_x, mask: test_mask, w: test_w, g: test_g}
        test_err = sess.run(loss, feed_dict = feed_dict)
        print('test_error:',test_err)



        

if __name__ == '__main__':
    main()
