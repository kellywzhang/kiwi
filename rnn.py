"""
Goal:
    - Create RNN layers

Important Concepts/Design Choices:
    - For "rnn" it is difficult in TF to iterate over variable number of iterations based
        on the value of a tensor (I couldn't figure this out, nor could I find any
        examples of others doing this). Thus how is it possible to iterate for variable
        number of time steps for each batch? This is where the use of TF's control
        flow options come in, namely tf.while_loop. See inline comments for details.

TODO/FIX: Get iteration numbers for RNN? Scope

Credits: Adapted from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/rnn.py
"""

import tensorflow as tf

def rnn(cell, inputs, seq_lens_mask, start_state=None):
    batch_size = tf.shape(inputs)[0]
    embedding_dim = inputs.get_shape().as_list()[2]

    if start_state == None:
        state = cell.zero_state(batch_size)
    else:
        state = start_state

    # Find the maximum document length, set as total number of time steps
    time = tf.cast(tf.reduce_max(tf.reduce_sum(seq_lens_mask, 1)), tf.int32)

    # Continue with loop if condition returns true
    def condition(i, inputs, state, outputs):
        return tf.less(i, time)

    # Body of while loop: runs one time step of RNN
    def body(i, inputs, state, outputs):
        # FIGURE OUT HOW TO GET PROPER NUMBERS HERE: with tf.variable_scope("Cell-Time{}".format(1)):
        with tf.variable_scope("Cell-Time"):
            # Take one time step's worth of input and create mask (time_mask discussed in rnn_cell.py)
            input_ = tf.slice(inputs, [0, i, 0], [batch_size, 1, embedding_dim])
            time_mask = tf.slice(seq_lens_mask, [0, i], [batch_size, 1])
            # Squeeze to get correct dimensions - dim 1 goes to 0
            input_ = tf.squeeze(input_, [1])

            # RNN time step
            output, state = cell(input_, state, time_mask)
            output = tf.expand_dims(output, 1)

            # Concatenate output to tensor of all outputs (hidden states)
            # Dimensions: batch x time x hidden_state_size
            # Concatenate along time
            outputs = tf.concat(1, [outputs, output])
        return [tf.add(i, 1), inputs, state, outputs]

    # iterator/counter
    i = tf.constant(0)

    # initialize "outputs" arg (hidden states) to pass into while loop to vector of zeros
    # Will remove these zeros after the while loop ends
    # Did this because need to concatenate current time step's hidden state with all prev
        # timestep's hidden states; can't concatenate with "None" as one argument
    outputs_shape = tf.TensorShape([None, None, cell._state_size])

    # Run RNN while loop
    _, _, last_state, hidden_states = tf.while_loop(condition, body, \
        loop_vars=[i, inputs, state, tf.zeros([batch_size, 1, cell._state_size])], \
        # Shape_invariants arg allows one to specify dimensions of inputs and outputs of each iterations
        # By using "None" as a dimension, signifies that this dimension can change
        shape_invariants=[i.get_shape(), inputs.get_shape(), state.get_shape(), outputs_shape])

    # get rid of zero output start state for concat purposes (see "body")
    hidden_states = tf.slice(hidden_states, [0, 1, 0], [batch_size, -1, cell._state_size])
    # reshape hidden_states to (batch x time x hidden_state_size)
    #hidden_states = tf.reshape(hidden_states, [batch_size, -1, cell._state_size])

    # Dimensions: outputs (batch x time x hidden_size); last_state (batch x hidden_size)
    return (hidden_states, last_state)


def bidirectional_rnn(forward_cell, backward_cell, inputs, seq_lens_mask, concatenate=True):
    # Reverse inputs (batch x time x embedding_dim); takes care of variable seq_len
    reverse_inputs = tf.reverse_sequence(inputs, tf.cast(tf.reduce_sum(seq_lens_mask, 1), tf.int32), seq_dim=1, batch_dim=0)

    # Run forwards and backwards RNN
    forward_outputs, forward_last_state = \
        rnn(forward_cell, inputs, seq_lens_mask)
    backward_outputs, backward_last_state = \
        rnn(backward_cell, reverse_inputs, seq_lens_mask)

    if concatenate:
        # last_state dimensions: batch x hidden_size
        last_state = tf.concat(1, [forward_last_state, backward_last_state])
        # outputs dimensions: batch x time x hidden_size
        outputs = tf.concat(2, [forward_outputs, backward_outputs])

        # Dimensions: outputs (batch x time x hidden_size*2); last_state (batch x hidden_size*2)
        return (outputs, last_state)

    # Dimensions: outputs (batch x time x hidden_size); last_state (batch x hidden_size)
    return (forward_outputs, forward_last_state, backward_outputs, backward_last_state)


def rnn_decoder(cell, start_state, inputs=None, seq_lens_mask=None, W_softmax=None, b_softmax=None, \
                W_embeddings=None, feed_previous=False, GO_ID=None, EOS_ID=None):
    """
    start_state is the vector to "decode"
    If feed_previous (greedy decoding):
        Requires: cell, start_state, W_embeddings, W_softmax (vocab_size x hidden_size), b_softmax, GO_ID, EOS_ID
    If not feed_previous
        Requires: cell, start_state, inputs, seq_lens_mask
    """

    if not feed_previous:
        # Decoding is equivalent to regular rnn if feeding in correct inputs (training)
        assert(inputs != None), "Must supply inputs if not decoding greedily"
        assert(seq_lens_mask != None), "Must supply seq_lens_mask if not decoding greedily"
        return rnn(cell, inputs, seq_lens_mask, start_state)

    batch_size = 1
    embedding_size = W_embeddings.get_shape().as_list()[1]
    #tf.shape(W_embeddings).as_list()[1]

    state = start_state
    GO_ID = tf.constant(GO_ID) # tf.Variable(GO_ID, trainable=False)
    EOS_ID = tf.constant(EOS_ID) #tf.Variable(EOS_ID, trainable=False)
    test = tf.constant(50)

    # Dimensions: batch x 1 x embedding_size
    embedding_seq = tf.gather(W_embeddings, [[GO_ID]])
    id_seq = tf.zeros([batch_size, 1], dtype=tf.int32)

    # Continue with loop if condition returns true
    def condition(i, embedding_seq, id_seq, state):
        # condition that data_utils.EOS_ID not predicted yet
        last_id = tf.squeeze(tf.slice(id_seq,[0,i],[batch_size,1]))

        return tf.logical_and(tf.less(i, test), tf.not_equal(last_id, EOS_ID))


    # Body of while loop: runs one time step of RNN
    def body(i, embedding_seq, id_seq, state):
        with tf.variable_scope("Cell-Time"):
            # Take one time step's worth of input and create mask (time_mask discussed in rnn_cell.py)
            time_mask = tf.ones([batch_size, 1])
            # Squeeze to get correct dimensions - dim 1 goes to 0
            input_ = tf.reshape(embedding_seq, [batch_size, embedding_size])

            # RNN time step
            output, state = cell(input_, state, time_mask)

            logits = tf.matmul(output, W_softmax, transpose_b=True) + b_softmax
            probabilities = tf.nn.softmax(logits)
            next_id = tf.cast(tf.argmax(probabilities, 1), tf.int32)
            next_id = tf.reshape(next_id, [1, 1])

            id_seq = tf.concat(1, [id_seq, next_id])

            # Concatenate output to tensor of all outputs (hidden states)
            # Dimensions: batch x time x hidden_state_size
            # Concatenate along time (reshape after while_loop finishes)

            embedding_seq = tf.gather(W_embeddings, next_id)
        return [tf.add(i, 1), embedding_seq, id_seq, state]

    # iterator/counter
    i = tf.constant(0)

    # initialize "outputs" arg (hidden states) to pass into while loop to vector of zeros
    # Will remove these zeros after the while loop ends
    # Did this because need to concatenate current time step's hidden state with all prev
        # timestep's hidden states; can't concatenate with "None" as one argument
    embedding_seq_shape = tf.TensorShape([None, None, embedding_size])
    #embedding_seq_shape = tf.TensorShape([embedding_size])
    id_seq_shape = tf.TensorShape([batch_size, None])

    # Run RNN while loop
    _, embedding_seq, id_seq, last_state = tf.while_loop(condition, body, \
        loop_vars=[i, embedding_seq, id_seq, state], \
        # Shape_invariants arg allows one to specify dimensions of inputs and outputs of each iterations
        # By using "None" as a dimension, signifies that this dimension can change
        shape_invariants=[i.get_shape(), embedding_seq_shape, id_seq_shape, state.get_shape()])
        #shape_invariants=[i.get_shape(), embedding_seq.get_shape(), id_seq.get_shape(), state.get_shape()])

    # get rid of zero output start state for concat purposes (see "body")
    id_seq = tf.slice(id_seq, [0, 1], [batch_size, -1])

    # Dimensions: outputs (batch x time x hidden_size); last_state (batch x hidden_size)
    return (id_seq, last_state)


def rnn_decoder_attention_forward(cell, start_state, attention, attended, W_softmax, b_softmax, W_embeddings, GO_ID, EOS_ID):
    batch_size = 1
    embedding_size = W_embeddings.get_shape().as_list()[1]

    state = start_state
    GO_ID = tf.constant(GO_ID)
    EOS_ID = tf.constant(EOS_ID)
    test = tf.constant(35)

    # Dimensions: batch x 1 x embedding_size
    embedding_seq = tf.gather(W_embeddings, [[GO_ID]])

    id_seq = tf.zeros([batch_size, 1], dtype=tf.int32)

    # Continue with loop if condition returns true
    def condition(i, embedding_seq, id_seq, state):
        # condition that data_utils.EOS_ID not predicted yet
        last_id = tf.squeeze(tf.slice(id_seq,[0,i],[batch_size,1]))

        return tf.logical_and(tf.less(i, test), tf.not_equal(last_id, EOS_ID))


    # Body of while loop: runs one time step of RNN
    def body(i, embedding_seq, id_seq, state):
        with tf.variable_scope("Cell-Time"):
            # Take one time step's worth of input and create mask (time_mask discussed in rnn_cell.py)
            time_mask = tf.ones([batch_size, 1])
            # Squeeze to get correct dimensions - dim 1 goes to 0
            input_ = tf.reshape(embedding_seq, [batch_size, embedding_size])

            #time_mask_attention = tf.ones_like(attended)
            alpha_weights, attend_result = \
                attention(attending=state, attended=attended)

            # batch x 1 x embedding -> batch x embedding
            context = tf.concat(1, [input_, attend_result])

            # RNN time step
            output, state = cell(context, state, time_mask)
            #output = tf.expand_dims(output, 1)

            logits = tf.matmul(output, W_softmax, transpose_b=True) + b_softmax
            probabilities = tf.nn.softmax(logits)
            next_id = tf.cast(tf.argmax(probabilities, 1), tf.int32)
            next_id = tf.reshape(next_id, [1, 1])

            id_seq = tf.concat(1, [id_seq, next_id])

            # Concatenate output to tensor of all outputs (hidden states)
            # Dimensions: batch x time x hidden_state_size
            # Concatenate along time (reshape after while_loop finishes)

            embedding_seq = tf.gather(W_embeddings, next_id)
        return [tf.add(i, 1), embedding_seq, id_seq, state]

    # iterator/counter
    i = tf.constant(0)

    # initialize "outputs" arg (hidden states) to pass into while loop to vector of zeros
    # Will remove these zeros after the while loop ends
    # Did this because need to concatenate current time step's hidden state with all prev
        # timestep's hidden states; can't concatenate with "None" as one argument
    embedding_seq_shape = tf.TensorShape([None, None, embedding_size])
    #embedding_seq_shape = tf.TensorShape([embedding_size])
    id_seq_shape = tf.TensorShape([batch_size, None])

    # Run RNN while loop
    _, embedding_seq, id_seq, last_state = tf.while_loop(condition, body, \
        loop_vars=[i, embedding_seq, id_seq, state], \
        # Shape_invariants arg allows one to specify dimensions of inputs and outputs of each iterations
        # By using "None" as a dimension, signifies that this dimension can change
        shape_invariants=[i.get_shape(), embedding_seq_shape, id_seq_shape, state.get_shape()])
        #shape_invariants=[i.get_shape(), embedding_seq.get_shape(), id_seq.get_shape(), state.get_shape()])

    # get rid of zero output start state for concat purposes (see "body")
    id_seq = tf.slice(id_seq, [0, 1], [batch_size, -1])

    # Dimensions: outputs (batch x time x hidden_size); last_state (batch x hidden_size)
    return (id_seq, last_state)

# cell size input size should be embedding_dim_input + embedding_dim_output
def rnn_decoder_attention(cell, start_state, inputs, inputs_mask, attentionf, attended):
    batch_size = tf.shape(inputs)[0]
    embedding_dim = inputs.get_shape().as_list()[2]

    if start_state == None:
        state = cell.zero_state(batch_size)
    else:
        state = start_state

    # Find the maximum document length, set as total number of time steps
    time = tf.cast(tf.reduce_max(tf.reduce_sum(inputs_mask, 1)), tf.int32)

    # Continue with loop if condition returns true
    def condition(i, inputs, state, outputs):
        return tf.less(i, time)

    # Body of while loop: runs one time step of RNN
    def body(i, inputs, state, outputs):
        with tf.variable_scope("Cell-Time"):
            # Take one time step's worth of input and create mask (time_mask discussed in rnn_cell.py)
            input_ = tf.slice(inputs, [0, i, 0], [batch_size, 1, embedding_dim])
            time_mask = tf.slice(inputs_mask, [0, i], [batch_size, 1])

            alpha_weights, attend_result = \
                attentionf(attending=state, attended=attended)

            # batch x 1 x embedding -> batch x embedding
            input_ = tf.squeeze(input_, [1])
            context = tf.concat(1, [input_, attend_result])

            # RNN time step
            output, state = cell(context, state, time_mask)
            output = tf.expand_dims(output, 1)

            # Concatenate output to tensor of all outputs (hidden states)
            # Dimensions: batch x time x hidden_state_size
            # Concatenate along time (reshape after while_loop finishes)
            outputs = tf.concat(1, [outputs, output])
        return [tf.add(i, 1), inputs, state, outputs]

    # iterator/counter
    i = tf.constant(0)

    # initialize "outputs" arg (hidden states) to pass into while loop to vector of zeros
    # Will remove these zeros after the while loop ends
    # Did this because need to concatenate current time step's hidden state with all prev
        # timestep's hidden states; can't concatenate with "None" as one argument
    outputs_shape = tf.TensorShape([None, None, cell._state_size])

    # Run RNN while loop
    _, _, last_state, hidden_states = tf.while_loop(condition, body, \
        loop_vars=[i, inputs, state, tf.zeros([batch_size, 1, cell._state_size])], \
        # Shape_invariants arg allows one to specify dimensions of inputs and outputs of each iterations
        # By using "None" as a dimension, signifies that this dimension can change
        shape_invariants=[i.get_shape(), inputs.get_shape(), state.get_shape(), outputs_shape])

    # get rid of zero output start state for concat purposes (see "body")
    hidden_states = tf.slice(hidden_states, [0, 1, 0], [batch_size, -1, cell._state_size])
    # reshape hidden_states to (batch x time x hidden_state_size)
    #hidden_states = tf.reshape(hidden_states, [batch_size, -1, cell._state_size])

    # Dimensions: outputs (batch x time x hidden_size); last_state (batch x hidden_size)
    return (hidden_states, last_state)
