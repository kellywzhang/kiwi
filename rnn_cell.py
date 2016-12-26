"""
Goal:
        - Create RNN cells

TODO:
        - Add peep-hole option to LSTMCell
        - Combine gate weights for efficiency

Credits: Adapted from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/rnn_cell.py
         Understanding from http://colah.github.io/posts/2015-08-Understanding-LSTMs/
"""

import tensorflow as tf
from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh

class RNNCell(object):
    """Abstract object representing an RNN cell."""

    def __call__(self, inputs, state, time_mask, scope=None):
        """Run this RNN cell on inputs, starting from the given state."""
        raise NotImplementedError("Abstract method")

    def zero_state(self, batch_size):
        """Return zero-filled state tensor(s)."""
        return tf.zeros(shape=[batch_size, self._state_size])

class GRUCell(RNNCell):
    """Gated Recurrent Unit (GRU) (http://arxiv.org/abs/1406.1078)."""

    def __init__(self, state_size, input_size, scope=None, activation=tanh):
        self._state_size = state_size
        self._input_size = input_size
        self._activation = activation
        self._scope = scope or type(self).__name__ # GRUCell

    def __call__(self, input_, hidden, state):
        # for GRU cell hidden and cell state are equivalent;
        # this signature is used for compatability with LSTMCell
        with tf.variable_scope(self._scope):
            input_size = self._input_size
            state_size = self._state_size
            inputs = tf.concat(1, [input_, state])

            with tf.variable_scope("Gates"): # Reset gate and update gate.
                # We start with bias of 1.0 to not reset and not update.
                self.W_reset = tf.get_variable(name="reset_weight", shape=[state_size+input_size, state_size], \
                        initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))
                self.W_update = tf.get_variable(name="update_weight", shape=[state_size+input_size, state_size], \
                        initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))
                self.b_reset = tf.get_variable(name="reset_bias", shape=[state_size], \
                        initializer=tf.constant_initializer(1.0))
                self.b_update = tf.get_variable(name="update_bias", shape=[state_size], \
                        initializer=tf.constant_initializer(1.0))

                reset_gate = sigmoid(tf.matmul(inputs, self.W_reset) + self.b_reset)
                update_gate = sigmoid(tf.matmul(inputs, self.W_update) + self.b_update)

            with tf.variable_scope("Candidate"):
                self.W_candidate = tf.get_variable(name="candidate_weight", shape=[state_size+input_size, state_size], \
                        initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))
                self.b_candidate = tf.get_variable(name="candidate_bias", shape=[state_size], \
                        initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))

                reset_input = tf.concat(1, [input_, reset_gate * state])
                candidate = self._activation(tf.matmul(reset_input, self.W_reset) + self.b_candidate)

                new_state = update_gate * state + (1 - update_gate) * candidate

        return new_state, new_state

    def zero_state(self, batch_size):
        return tf.cast(tf.zeros([batch_size, self._state_size]), dtype=tf.float32)

class LSTMCell(RNNCell):
    """Long Short Term Memory (LSTM) Cell (http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf)."""

    def __init__(self, state_size, input_size, scope=None, activation=tanh):
        self._state_size = state_size     # cell size and hidden state size equivalent
        self._input_size = input_size
        self._activation = activation
        self._scope = scope or type(self).__name__ # "LSTMCell"

    def __call__(self, input_, hidden, state):
        # state is previous cell state
        with tf.variable_scope(self._scope):
            input_size = self._input_size
            state_size = self._state_size
            inputs = tf.concat(1, [input_, hidden])

            with tf.variable_scope("Gates"):
                self.W_forget = tf.get_variable(name="forget_weight", shape=[state_size+input_size, state_size], \
                        initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))
                self.W_input = tf.get_variable(name="input_weight", shape=[state_size+input_size, state_size], \
                        initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))
                self.W_output = tf.get_variable(name="output_weight", shape=[state_size+input_size, state_size], \
                        initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))

                # We start with bias of 1.0 to not forget.
                self.b_forget = tf.get_variable(name="forget_bias", shape=[state_size], \
                        initializer=tf.constant_initializer(1.0))
                self.b_input = tf.get_variable(name="input_bias", shape=[state_size], \
                        initializer=tf.constant_initializer(0.0))
                self.b_output = tf.get_variable(name="output_bias", shape=[state_size], \
                        initializer=tf.constant_initializer(0.0))

                forget_gate = sigmoid(tf.matmul(inputs, self.W_forget) + self.b_forget)
                input_gate = sigmoid(tf.matmul(inputs, self.W_input) + self.b_input)
                output_gate = sigmoid(tf.matmul(inputs, self.W_output) + self.b_output)

            with tf.variable_scope("Candidate"):
                self.W_candidate = tf.get_variable(name="candidate_weight", shape=[state_size+input_size, state_size], \
                        initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))
                self.b_candidate = tf.get_variable(name="candidate_bias", shape=[state_size], \
                        initializer=tf.constant_initializer(0.0))

                candidate = self._activation(tf.matmul(inputs, self.W_candidate) + self.b_candidate)
                new_state = forget_gate * state + input_gate * candidate
                new_output = output_gate * self._activation(new_state)

        return new_output, new_state

    def zero_state(self, batch_size):
        return tf.cast(tf.zeros([batch_size, self._state_size]), dtype=tf.float32)
