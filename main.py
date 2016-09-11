""" Author: Zander Blasingame
    Location: CAMEL at Clarkson University
    Purpose: MLP for the purpose of recognizing malicious
            hardware requests.
    Documentation: Enter `python main.py --help` """

import argparse
import numpy as np
import tensorflow as tf

from utils import parse_csv
from utils import one_hot_encoding
from MLP import MLP

# Parse Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--train',
                    action='store_true',
                    help='Flag to train neural net on dataset')
parser.add_argument('--testing',
                    action='store_true',
                    help='Flag to turn on testing')
parser.add_argument('--train_file',
                    type=str,
                    help='Location of training file')
parser.add_argument('--test_file',
                    type=str,
                    help='Location of testing file')
parser.add_argument('--parser_stats',
                    action='store_true',
                    help='Flag to print results in a parser friendly format')
parser.add_argument('--normalize',
                    action='store_true',
                    help='Flag to normalize input data')

args = parser.parse_args()

mode = args.mode

normalize = False if not args.normalize else True
if args.train:
    trX, trY = parse_csv(args.train_file, num_hpc=12, normalize=normalize)
    trY = one_hot_encoding(trY)

if args.testing:
    teX, teY = parse_csv(args.test_file, num_hpc=12, normalize=normalize)
    teY = one_hot_encoding(teY)


# Network parameters
learning_rate = 0.001
reg_param = 0.01
dropout_prob = 0.5
training_epochs = 4
display_step = 1
std_pram = 1.0
num_input = len(trX[0]) if args.train else len(teX[0])
num_units = 15 if args.num_units is None else args.num_units
num_out = len(trY[0]) if args.train else len(teY[0])
training_size = len(trX) if args.train else None
testing_size = len(teX) if args.testing else None

# Placeholders
X = tf.placeholder('float', [None, num_input])
Y = tf.placeholder('float', [None, num_out])
keep_prob = tf.placeholder('float')

# Create Network
model_name = 'MLP Classifier'
mlp = MLP([num_input, num_units, num_out],
          [tf.nn.relu, tf.identity])

logits = mlp.create_network(X, keep_prob)

cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, Y)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(predictions, 'float'))

init_op = tf.initialize_all_variables()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init_op) if args.train else saver.restore(sess, 'model.ckpt')

    if args.train:
        for epoch in range(training_epochs):
            avg_cost = 0
            for i in range(training_size):
                if trY[i] == 1:
                    feed_dict = {X: trX[i],
                                 Y: trY[i],
                                 keep_prob: dropout_prob}

                    _, c = sess.run([optimizer, cost],
                                    feed_dict=feed_dict)

                    avg_cost += c / training_size

            if i % display_step == 0:
                print('Epoch: {0:03} with cost={1:.9f}'.format(epoch+1,
                                                               avg_cost))

        print('Optimization Finished')

        # save model
        save_path = saver.save(sess, 'model.ckpt')
        print('Model saved in file: {}'.format(save_path))

    if args.testing:
        if args.parser_stats:
            print('PARSER_STATS_BEGIN')

        print('model_name={}'.format(model_name))
        print('accuracy={}'.format(sess.run(accuracy,
                                            feed_dict={X: teX,
                                                       Y: teY,
                                                       keep_prob: 1.0})))
