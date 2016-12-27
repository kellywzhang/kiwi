"""
Goal:
    - Create RNN layers

Important Concepts/Design Choices:
    - I wanted to implement dynamic unrolling (don't have to pick a set cutoff length).
        I utilized TF's control flow option tf.while_loop to make it possible to iterate
        over a veriable number of time steps for each batch. See inline comments for details.

TODO:
    - Make multi-layer RNN option

Credits: Adapted from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/rnn.py
         Understanding from http://colah.github.io/posts/2015-08-Understanding-LSTMs/
"""

import tensorflow as tf
try:
    import math_ops
except ImportError:
    from kiwi import math_ops

def rnn(cell, inputs, seq_lens, start_state=None):
    batch_size = tf.shape(inputs)[0]
    embedding_dim = tf.shape(inputs)[2]

    # start state is all zeros unless one is provided
    if start_state is not None:
        tf.assert_equal(batch_size, tf.shape(start_state)[0])
        tf.assert_equal(cell._state_size, tf.shape(start_state)[1])
    else:
        start_state = cell.zero_state(batch_size)

    # Find the maximum document length, set as total number of time steps
    time = tf.cast(tf.reduce_max(seq_lens), tf.int32)

    # Continue with loop if condition returns true
    def condition(i, inputs, state, outputs):
        return tf.less(i, time)

    # Body of while loop: runs one time step of RNN
    def body(i, inputs, state, outputs):
        # one RNN time step
        with tf.variable_scope("Cell-Time"):
            # Take one time step's worth of input; then squeeze to get correct dimensions - dim 1 goes to 0
            input_ = tf.slice(inputs, [0, i, 0], [batch_size, 1, embedding_dim])
            input_ = tf.squeeze(input_, [1])
            # last timestep's output (hidden state)
            last_output = tf.slice(outputs, [0, i, 0], [batch_size, 1, cell._state_size])
            last_output = tf.squeeze(last_output, [1])

            # Dimensions of output == Dimensions of new_state, (batch_size x state_size)
            output, new_state = cell(input_, last_output, state)
            # only allow sequences that are long enough to pass through
            time_mask = i < seq_lens
            output = math_ops.cond_gather(time_mask, true_tensor=output, false_tensor=last_output)

            # Concatenate output to tensor of all outputs (hidden states) along time dimension
            # Dimensions: batch x time x hidden_state_size
            output = tf.expand_dims(output, 1)
            outputs = tf.concat(1, [outputs, output])
        return [tf.add(i, 1), inputs, new_state, outputs]

    # iterator/counter
    i = tf.constant(0)

    # initialize "outputs" arg (hidden states) to pass into while loop to vector of zeros
    # Will remove these zeros after the while loop ends
    # Did this because need to concatenate current time step's hidden state with all prev
        # timestep's hidden states; can't concatenate with "None" as one argument
    outputs_shape = tf.TensorShape([None, None, cell._state_size])

    # Run RNN while loop
    _, _, last_state, hidden_states = tf.while_loop(condition, body, \
        loop_vars=[i, inputs, start_state, tf.zeros([batch_size, 1, cell._state_size])], \
        # Shape_invariants arg allows one to specify dimensions of inputs and outputs of each iterations
        # By using "None" as a dimension, signifies that this dimension can change
        shape_invariants=[i.get_shape(), inputs.get_shape(), start_state.get_shape(), outputs_shape])

    # get rid of zero output start state for concat purposes (see "body")
    hidden_states = tf.slice(hidden_states, [0, 1, 0], [batch_size, -1, cell._state_size])

    # Dimensions: outputs (batch x time x hidden_size); last_state (batch x hidden_size)
    return (hidden_states, last_state)


def bidirectional_rnn(forward_cell, backward_cell, inputs, seq_lens, concatenate=True):

    # Reverse inputs (batch x time x embedding_dim); takes care of variable seq_len
    reverse_input = tf.reverse_sequence(inputs, seq_lens, seq_dim=1, batch_dim=0)

    # Run forwards and backwards RNN
    forward_outputs, forward_last_state = rnn(forward_cell, inputs, seq_lens)
    backward_outputs_reversed, backward_last_state = rnn(backward_cell, reverse_input, seq_lens)

    # reverse outputs
    backward_outputs = tf.reverse_sequence(backward_outputs_reversed, seq_lens, seq_dim=1, batch_dim=0)

    if concatenate:
        # last_state dimensions: batch x hidden_size
        last_state = tf.concat(1, [forward_last_state, backward_last_state])
        # outputs dimensions: batch x time x hidden_size
        outputs = tf.concat(2, [forward_outputs, backward_outputs])

        # Dimensions: outputs (batch x time x hidden_size*2); last_state (batch x hidden_size*2)
        return (outputs, last_state)

    # Dimensions: outputs (batch x time x hidden_size); last_state (batch x hidden_size)
    return (forward_outputs, forward_last_state, backward_outputs, backward_last_state)



# def multilayer_rnn(cells, inputs, seq_lens, dropout=True, dropout_keep_prob=0.8):
#     # pass in a list of cells
#     for i, cell in enumerate(cells):
#     return
#
# class MultiRNNCell(RNNCell):
#   """RNN cell composed sequentially of multiple simple cells."""
#
#   def __init__(self, cells, state_is_tuple=True):
#     """Create a RNN cell composed sequentially of a number of RNNCells.
#
#     Args:
#       cells: list of RNNCells that will be composed in this order.
#       state_is_tuple: If True, accepted and returned states are n-tuples, where
#         `n = len(cells)`.  If False, the states are all
#         concatenated along the column axis.  This latter behavior will soon be
#         deprecated.
#
#     Raises:
#       ValueError: if cells is empty (not allowed), or at least one of the cells
#         returns a state tuple but the flag `state_is_tuple` is `False`.
#     """
#     if not cells:
#       raise ValueError("Must specify at least one cell for MultiRNNCell.")
#     self._cells = cells
#     self._state_is_tuple = state_is_tuple
#     if not state_is_tuple:
#       if any(nest.is_sequence(c.state_size) for c in self._cells):
#         raise ValueError("Some cells return tuples of states, but the flag "
#                          "state_is_tuple is not set.  State sizes are: %s"
#                          % str([c.state_size for c in self._cells]))
#
#   @property
#   def state_size(self):
#     if self._state_is_tuple:
#       return tuple(cell.state_size for cell in self._cells)
#     else:
#       return sum([cell.state_size for cell in self._cells])
#
#   @property
#   def output_size(self):
#     return self._cells[-1].output_size
#
#   def __call__(self, inputs, state, scope=None):
#     """Run this multi-layer cell on inputs, starting from state."""
#     with vs.variable_scope(scope or "multi_rnn_cell"):
#       cur_state_pos = 0
#       cur_inp = inputs
#       new_states = []
#       for i, cell in enumerate(self._cells):
#         with vs.variable_scope("cell_%d" % i):
#           if self._state_is_tuple:
#             if not nest.is_sequence(state):
#               raise ValueError(
#                   "Expected state to be a tuple of length %d, but received: %s"
#                   % (len(self.state_size), state))
#             cur_state = state[i]
#           else:
#             cur_state = array_ops.slice(
#                 state, [0, cur_state_pos], [-1, cell.state_size])
#             cur_state_pos += cell.state_size
#           cur_inp, new_state = cell(cur_inp, cur_state)
#           new_states.append(new_state)
#     new_states = (tuple(new_states) if self._state_is_tuple
#                   else array_ops.concat(1, new_states))
#     return cur_inp, new_states
