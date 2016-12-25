import tensorflow as tf
import numpy as np
import rnn_cell, rnn

class RNNClassifier(object):
    """
    Purpose:
    Instances of this class
    """
    def __init__(self, num_classes, vocab_size, hidden_size=128, \
        embedding_dim=100, batch_size=32, bidirectional=False):

        tf.set_random_seed(1234)

        # Placeholders
        # can add assert statements to ensure shared None dimensions are equal (batch_size)
        self.seq_lens = tf.placeholder(tf.int32, [None, ], name="seq_lens")
        self.input_x = tf.placeholder(tf.int32, [None, None], name="input_x")
        self.input_y = tf.placeholder(tf.int32, [None, ], name="input_y")

        mask_x = tf.cast(tf.sequence_mask(self.seq_lens), tf.int32)

        # Document and Query embeddings; One-hot-encoded answers
        masked_x = tf.mul(self.input_x, mask_x)
        one_hot_y = tf.one_hot(self.input_y, num_classes)

        # Buildling Graph (Network Layers)
        # ==================================================
        with tf.variable_scope("embedding"):
            self.W_embeddings = tf.get_variable(shape=[vocab_size, embedding_dim], \
                                           initializer=tf.random_uniform_initializer(-0.01, 0.01),\
                                           name="W_embeddings")

            # Dimensions: batch x max_length x embedding_dim
            input_embedding = tf.gather(self.W_embeddings, masked_x)

        with tf.variable_scope("rnn"):
            if bidirectional:
                # Bidirectional RNNs
                forward_cell = rnn_cell.GRUCell(state_size=hidden_size, input_size=embedding_dim, scope="GRU-Forward")
                backward_cell = rnn_cell.GRUCell(state_size=hidden_size, input_size=embedding_dim, scope="GRU-Backward")

                hidden_states, last_state = rnn.bidirectional_rnn(forward_cell, backward_cell, \
                    input_embedding, self.seq_lens, batch_size, embedding_dim, concatenate=True)
            else:
                # One directional RNN (start to end)
                cell = rnn_cell.GRUCell(state_size=hidden_size, input_size=embedding_dim, scope="GRU")
                hidden_states, last_state = rnn.rnn(cell, input_embedding, self.seq_lens, \
                    batch_size, embedding_dim)

        with tf.variable_scope("prediction"):
            if bidirectional:
                W_predict = tf.get_variable(name="predict_weight", shape=[hidden_size*2, num_classes], \
                    initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))
            else:
                W_predict = tf.get_variable(name="predict_weight", shape=[hidden_size, num_classes], \
                    initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))
            b_predict = tf.get_variable(name="predict_bias", shape=[num_classes],
                initializer=tf.constant_initializer(0.0))
            # Dimensions (batch_size x num_classes)
            prediction_probs_unnormalized = tf.matmul(last_state, W_predict) + b_predict

            # Softmax
            # Dimensions (batch x time)
            prediction_probs = tf.nn.softmax(prediction_probs_unnormalized, name="prediction_probs")
            likelihoods = tf.reduce_sum(tf.mul(prediction_probs, one_hot_y), 1)
            log_likelihoods = tf.log(likelihoods)

            # Negative log-likelihood loss
            self.loss = tf.mul(tf.reduce_sum(log_likelihoods), -1)
            predictions = tf.argmax(prediction_probs, 1, name="predictions")
            correct_vector = tf.cast(tf.equal(tf.argmax(one_hot_y, 1), tf.argmax(prediction_probs, 1)), \
                tf.float32, name="correct_vector")
            self.accuracy = tf.reduce_mean(correct_vector)
