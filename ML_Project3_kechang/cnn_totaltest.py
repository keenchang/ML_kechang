import tensorflow as tf
import numpy as np
import pandas as pd
import os


# label(y_i -> bit label)
def label_trans_bit(data, i, j):
    a1 = str(data.iloc[i, 0])
    b1 = str(data.iloc[i, 1])
    a2 = str(data.iloc[i, 2])
    b2 = str(data.iloc[i, 3])
    c = int((a1+b1+a2+b2), 16)
    d = '{:016b}'.format(c)
    e = int(d[j])
    return e


# label(y -> total label)
def label_trans(data, i):
    a1 = str(data.iloc[i, 0])
    b1 = str(data.iloc[i, 1])
    a2 = str(data.iloc[i, 2])
    b2 = str(data.iloc[i, 3])
    c = int((a1+b1+a2+b2), 16)
    d = '{:016b}'.format(c)
    return d


# product [x_train, y_train, x_test, y_test] by [rawdata, x_2bit, shuffled, labelbit]
def preprocessing(raw_data, x_2bit, shuffled, label_bit, test_percentage=0.3):
    # read label(y)-(0 or 1 for each bit)
    ya = []
    for k in range(len(raw_data)):
        yf = label_trans_bit(raw_data, k, label_bit)
        ya.append(yf)
    y = np.array(ya, dtype=np.int32)

    # Randomly shuffle data
    x_shuffled = x_2bit[shuffled]
    y_shuffled = y[shuffled]

    # split data
    test_sample_index = -1 * int(test_percentage * float(len(y)))
    x_train, x_test = x_shuffled[:test_sample_index], x_shuffled[test_sample_index:]
    y_train, y_test = y_shuffled[:test_sample_index], y_shuffled[test_sample_index:]

    del x_2bit, y, x_shuffled, y_shuffled
    return x_train, y_train, x_test, y_test


# product model, train model and output prediction, accuracy of each bit
def model(xtrain, ytrain, xtest, ytest, label_bit, epoch=100, dropout_keep_prob=0.5):

    checkname = str(label_bit) + "bit"
    # graph(CNN model)
    g = tf.Graph()
    with g.as_default():

        # placeholder
        input_x = tf.placeholder(tf.float32, [None, 28, 4], name="input_x")
        input_y = tf.placeholder(tf.int32, [None], name="input_y")
        dropout_prob = tf.placeholder(tf.float32, name="dropout_prob")

        # let feature be 4D and label be one hot label
        input_expanded = tf.expand_dims(input_x, axis=-1)
        one_hot_label = tf.one_hot(input_y, 2, axis=1, name='one_hot_label')

        # conv_1
        filter_shape1 = [3, 4, 1, 512]
        conv_W1 = tf.Variable(tf.random_normal(filter_shape1, mean=0.0, stddev=0.01), name="conv_W1")
        conv_b1 = tf.Variable(tf.constant(0.01, shape=[512]), name="conv_b1")
        conv1 = tf.nn.conv2d(input_expanded, conv_W1,
                             strides=[1, 1, 1, 1], padding="VALID", name="conv1")
        act1 = tf.nn.relu(tf.nn.bias_add(conv1, conv_b1), name="act1")
        pool1 = tf.nn.max_pool(act1, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='VALID', name="pool1")

        # reshape output of convolution layers
        reshap = tf.reshape(pool1, [-1, 13*512], name='reshap')

        # Affine layer(fully connected layer)
        W_f = tf.Variable(tf.random_normal([13*512, 640]) * tf.sqrt(2.0 / (13.0*512.0)), name="W_f")
        b_f = tf.Variable(tf.random_normal([640]) * tf.sqrt(2.0 / (13.0*512.0)), name="b_f")
        ful = tf.matmul(reshap, W_f) + b_f
        fully = tf.nn.relu(ful, name="fully")

        # Add dropout
        dropout = tf.layers.dropout(fully, dropout_prob)

        # Classification - output and predictions
        W_c = tf.Variable(tf.random_normal([640, 2]) * tf.sqrt(2.0/640.0), name="W_c")
        b_c = tf.Variable(tf.random_normal([2]) * tf.sqrt(2.0/640.0), name="b_c")
        out = tf.matmul(dropout, W_c) + b_c
        output = tf.nn.softmax(out, axis=1, name="output")
        predictions = tf.argmax(output, 1, name="predictions")

        # Calculate mean cross-entropy loss
        losses = -tf.reduce_sum(one_hot_label * tf.log(output + 1e-15), reduction_indices=1)
        # regular = tf.nn.l2_loss(self.W1) + tf.nn.l2_loss(self.W2) + tf.nn.l2_loss(self.W)
        loss = tf.reduce_mean(losses)  # + 0.005*regular

        # Accuracy
        correct_predictions = tf.equal(predictions, tf.argmax(one_hot_label, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        # Optimizer
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
        train_op = optimizer.minimize(loss, name='train_op', global_step=global_step)

        # save model
        checkpoint_dir = os.path.abspath(os.path.join(os.path.curdir, "totaltest_runs", checkname))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables())

    ##########################################################
    # session(train model and save model)
    allow_soft_placement = True
    log_device_placement = False
    session_conf = tf.ConfigProto(allow_soft_placement=allow_soft_placement,
                                  log_device_placement=log_device_placement)
    with tf.Session(graph=g, config=session_conf) as sess:
        sess.run(tf.global_variables_initializer())
        # train model
        for m in range(epoch):
            random_indices = np.random.permutation(np.arange(len(ytrain)))
            train_feature = xtrain[random_indices]
            train_label = ytrain[random_indices]
            for n in range(3600):
                feature = train_feature[32*n: 32*n + 32]
                label = train_label[32*n: 32*n + 32]

                feed_dict = {input_x: feature, input_y: label, dropout_prob: dropout_keep_prob}
                _, step, loss_, accuracy_ = sess.run(
                    [train_op, global_step, loss, accuracy], feed_dict)
                print("bit: %i, step: %i, loss: %.4f, acc: %g" % (label_bit, step, loss_, accuracy_))
                current_step = tf.train.global_step(sess, global_step)
            print("\n Evaluation:")
            feed_dict = {input_x: xtest, input_y: ytest, dropout_prob: 0.0}
            step, loss_, accuracy_ = sess.run([global_step, loss, accuracy], feed_dict)
            print("bit: %i, epoch: %i, loss: %.4f, acc: %g" % (label_bit, m, loss_, accuracy_))
            print("")
        feed_dict = {input_x: xtest, input_y: ytest, dropout_prob: 0.0}
        predict_bit = sess.run(predictions, feed_dict)
        acc_bit = sess.run(accuracy, feed_dict)
    return predict_bit, acc_bit


##############################################################################
# Load data
rawdata = pd.read_csv('./project_data.csv')

feature = pd.read_csv('./project_feature.csv', header=None)
x_ = np.array(feature.values, dtype=np.float32)
x_2bit = np.reshape(x_, [-1, 28, 4])

shuffle_indices = np.array(pd.read_csv('./shuffle_index.csv'), dtype=np.int32)
shuffle_indices = shuffle_indices.flatten()

# each bit test for i=0~15
predict_bit_list = []
true_bit_list = []
acc_bit_list = []

for i in range(16):
    xtrain_bit, ytrain_bit, xtest_bit, ytest_bit = preprocessing(raw_data=rawdata, x_2bit=x_2bit, 
                                                                 shuffled=shuffle_indices, label_bit=i, 
                                                                 test_percentage=0.3)
    predict_bit, acc_bit = model(xtrain=xtrain_bit, ytrain=ytrain_bit, xtest=xtest_bit, ytest=ytest_bit, 
                                label_bit=i, epoch=100)
    predict_bit_ = np.array(predict_bit.flatten(), dtype=np.int32)
    predict_bit_list.append(predict_bit_)
    true_bit_list.append(ytest_bit)
    acc_bit_list.append(acc_bit)

# prediction for each bit(i=0~15) with x_test, y_test
predict_bit0 = predict_bit_list[0]
predict_bit1 = predict_bit_list[1]
predict_bit2 = predict_bit_list[2]
predict_bit3 = predict_bit_list[3]
predict_bit4 = predict_bit_list[4]
predict_bit5 = predict_bit_list[5]
predict_bit6 = predict_bit_list[6]
predict_bit7 = predict_bit_list[7]
predict_bit8 = predict_bit_list[8]
predict_bit9 = predict_bit_list[9]
predict_bit10 = predict_bit_list[10]
predict_bit11 = predict_bit_list[11]
predict_bit12 = predict_bit_list[12]
predict_bit13 = predict_bit_list[13]
predict_bit14 = predict_bit_list[14]
predict_bit15 = predict_bit_list[15]

# true for each bit(i=0~15) with x_test, y_test
true_bit0 = true_bit_list[0]
true_bit1 = true_bit_list[1]
true_bit2 = true_bit_list[2]
true_bit3 = true_bit_list[3]
true_bit4 = true_bit_list[4]
true_bit5 = true_bit_list[5]
true_bit6 = true_bit_list[6]
true_bit7 = true_bit_list[7]
true_bit8 = true_bit_list[8]
true_bit9 = true_bit_list[9]
true_bit10 = true_bit_list[10]
true_bit11 = true_bit_list[11]
true_bit12 = true_bit_list[12]
true_bit13 = true_bit_list[13]
true_bit14 = true_bit_list[14]
true_bit15 = true_bit_list[15]

# let 16 prediction and true be combined to 1 prediction and true
sum_ = 0

for j in range(len(predict_bit0)):
    pre = str(predict_bit0[j]) + str(predict_bit1[j]) + str(predict_bit2[j]) + str(predict_bit3[j]) + \
          str(predict_bit4[j]) + str(predict_bit5[j]) + str(predict_bit6[j]) + str(predict_bit7[j]) + \
          str(predict_bit8[j]) + str(predict_bit9[j]) + str(predict_bit10[j]) + str(predict_bit11[j]) + \
          str(predict_bit12[j]) + str(predict_bit13[j]) + str(predict_bit14[j]) + str(predict_bit15[j])

    true_ = str(true_bit0[j]) + str(true_bit1[j]) + str(true_bit2[j]) + str(true_bit3[j]) + \
            str(true_bit4[j]) + str(true_bit5[j]) + str(true_bit6[j]) + str(true_bit7[j]) + \
            str(true_bit8[j]) + str(true_bit9[j]) + str(true_bit10[j]) + str(true_bit11[j]) + \
            str(true_bit12[j]) + str(true_bit13[j]) + str(true_bit14[j]) + str(true_bit15[j])

    if pre == true_:
        sum_ = sum_ + 1
total_acc = sum_/len(predict_bit0)

# print accuracy of each bit and total accuracy
for p in range(16):
    print('Accuracy of '+str(p)+'bit for testing: %.4f' % (acc_bit_list[p]))
print('Total accuracy of y for testing: %.4f' % total_acc)




