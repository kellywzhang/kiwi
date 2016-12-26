import tensorflow as tf
import numpy as np
import datetime

from kiwi import rnn_cell, rnn
import tf_helpers

class Seq2Seq(object):
    def __init__(self, hidden_size=128, vocab_size=50000, embedding_size=100, \
                 batch_size=32, random_seed=1234, forward_only=False, bidirectional=False):

        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        tf.set_random_seed(random_seed)

        # EncoderDecoder -----------------------------------------
        # Placeholders
        self.source_sent = tf.placeholder(tf.int32, [None, None], name="source_sent")
        self.target_sent = tf.placeholder(tf.int32, [None, None], name="target_sent")
        self.target_sent_nogo = tf.placeholder(tf.int32, [None, None], name="target_sent_nogo")

        # length of target sent
        #time = tf.shape(self.target_sent)[1]

        # Dynamically calculate length of each example (Dimension: batch_size)
        # Assumes sequences are padded with negative numbers
        source_lens = tf.reduce_sum(tf.cast(self.source_sent >= 0, tf.int32), 1)
        target_lens = tf.reduce_sum(tf.cast(self.target_sent >= 0, tf.int32), 1)

        # Create sequence_masks
        source_mask = tf.sequence_mask(source_lens, dtype=tf.int32)
        target_mask = tf.sequence_mask(target_lens, dtype=tf.int32)

        masked_source = tf.mul(source_mask, self.source_sent)
        masked_target = tf.mul(target_mask, self.target_sent)
        masked_target_nogo = tf.mul(target_mask, self.target_sent_nogo)

        # Create sequence_masks
        source_mask = tf.cast(source_mask, dtype=tf.float32)
        target_mask = tf.cast(target_mask, dtype=tf.float32)

        with tf.variable_scope("embedding"):
            self.W_embeddings_source = tf.get_variable(shape=[vocab_size, embedding_size], \
                                               initializer=tf.random_uniform_initializer(-0.01, 0.01),\
                                               name="W_embeddings_source")

            # Dimensions: batch x max_length x embedding_dim
            source_embedding = tf.gather(self.W_embeddings_source, masked_source)

            self.W_embeddings_target = tf.get_variable(shape=[vocab_size, embedding_size], \
                                               initializer=tf.random_uniform_initializer(-0.01, 0.01),\
                                               name="W_embeddings_target")

            # Dimensions: batch x max_length x embedding_dim
            target_embedding = tf.gather(self.W_embeddings_target, masked_target)

        with tf.variable_scope("encoder"):
            if bidirectional:
                # One directional RNN (start to end)
                forward_cell = rnn_cell.GRUCell(state_size=int(hidden_size/2), input_size=embedding_size, scope="GRU_encoder_forward")
                backward_cell = rnn_cell.GRUCell(state_size=int(hidden_size/2), input_size=embedding_size, scope="GRU_encoder_backward")

                hidden_states_encoder, self.last_state_encoder = \
                    rnn.bidirectional_rnn(forward_cell, backward_cell, source_embedding, source_mask, concatenate=True)
            else:
                # One directional RNN (start to end)
                encoder_cell = rnn_cell.GRUCell(state_size=hidden_size, input_size=embedding_size, scope="GRU_encoder")
                hidden_states_encoder, self.last_state_encoder = rnn.rnn(encoder_cell, source_embedding, source_mask)


            # Add projection layer between encoder and decoder?
            self.W_projection = tf.get_variable(shape=[hidden_size, hidden_size], \
                                           initializer=tf.random_uniform_initializer(-0.01, 0.01),\
                                           name="W_projection")
            # batch_size x hidden_size; hidden_size x hidden_size
            self.last_state_encoder = tf.matmul(self.last_state_encoder, self.W_projection)


        with tf.variable_scope("decoder"):
            self.W_softmax = tf.get_variable(shape=[vocab_size, hidden_size], \
                                           initializer=tf.random_uniform_initializer(-0.01, 0.01),\
                                           name="W_softmax")
            self.b_softmax = tf.get_variable(name="softmax_bias", shape=[vocab_size], \
                initializer=tf.constant_initializer(0.0))

            self.decoder_cell = rnn_cell.GRUCell(state_size=hidden_size, input_size=embedding_size, scope="GRU_decoder")

            if not forward_only:
                hidden_states_decoder, last_state_decoder = \
                    rnn.rnn_decoder(cell=self.decoder_cell,
                                    start_state=self.last_state_encoder,
                                    inputs=target_embedding,
                                    seq_lens_mask=target_mask)
            else:
                self.id_seq, last_state = \
                    rnn.rnn_decoder(cell=self.decoder_cell,
                                    start_state=self.last_state_encoder,
                                    W_softmax=self.W_softmax,
                                    b_softmax=self.b_softmax,
                                    W_embeddings=self.W_embeddings_target,
                                    feed_previous=True,
                                    GO_ID=data_utils.GO_ID,
                                    EOS_ID=data_utils.EOS_ID)

        if not forward_only:
            with tf.variable_scope("loss"):
                # User float32 to avoid numerical instability with sampled_softmax_loss
                #masked_target = tf.slice(masked_target, [0, 1], [batch_size, time-1])
                ### HEREEHEERERHE
                labels = tf.reshape(tf.boolean_mask(masked_target_nogo, tf.cast(target_mask, dtype=tf.bool)), [-1, 1])
                hidden_states = tf.boolean_mask(hidden_states_decoder, tf.cast(target_mask, dtype=tf.bool))

                # calculate loss
                loss_vector = tf.nn.sampled_softmax_loss(
                    weights=self.W_softmax,
                    biases=self.b_softmax,
                    inputs=hidden_states,
                    labels=labels,
                    num_sampled=600,
                    num_classes=vocab_size,
                    num_true=1,
                    remove_accidental_hits=True,
                    name='sampled_softmax_loss'
                )
                # average across timesteps
                total_timesteps = tf.reduce_sum(target_lens)
                # loss is equal to log-perplexity
                self.loss = tf.div(tf.reduce_sum(loss_vector), \
                    tf.cast(total_timesteps, tf.float32), name="loss")
                self.perplexity = tf.exp(self.loss, name="perplexity")

        self.saver = tf.train.Saver(tf.all_variables())


    def set_train(self, session, max_global_norm, optimizer, FLAGS, timestamp=None):
        # Define Training procedure
        trainables = tf.trainable_variables()
        grads = tf.gradients(self.loss, trainables)
        grads, _ = tf.clip_by_global_norm(grads, clip_norm=max_global_norm)
        grad_var_pairs = zip(grads, trainables)

        #print(grads)
        self.train_op = optimizer.apply_gradients(grad_var_pairs, global_step=self.global_step)
        #grads_and_vars = optimizer.compute_gradients(model.loss)
        #print(grads_and_vars)
        #clipped_gvs = tf.clip_by_global_norm(tf.gradients(loss, tvars), 1)
        #clipped_gvs = [(tf.clip_by_value(grad, -FLAGS.grad_clip, FLAGS.grad_clip), var) for grad, var in grads_and_vars]
        #train_op = optimizer.apply_gradients(grads_and_vars, global_step=model.global_step)

        # Summaries for loss and perplexity
        loss_summary = tf.scalar_summary("loss", self.loss)
        perp_summary = tf.scalar_summary("perplexity", self.perplexity)

        self.train_summary_op, self.dev_summary_op, self.train_summary_writer, \
            self.dev_summary_writer, self.timestamp, self.checkpoint_prefix = \
            tf_helpers.save_summaries(session, [loss_summary, perp_summary], grad_var_pairs, FLAGS, timestamp)


    def step(self, session, encoder_inputs, decoder_inputs, decoder_nogo, train=True, print_bool=False, forward_only=False, graph=None):
        feed_dict = {
            self.source_sent: encoder_inputs,
            self.target_sent: decoder_inputs,
            self.target_sent_nogo: decoder_nogo
        }
        time_str = datetime.datetime.now().isoformat()

        if not forward_only:
            if train:
                _, step, summaries, loss, perplexity = session.run(
                    [self.train_op, self.global_step, self.train_summary_op, self.loss, self.perplexity],
                    feed_dict)
                self.train_summary_writer.add_summary(summaries, step)
            else:
                step, summaries, loss, perplexity = session.run(
                    [self.global_step, self.dev_summary_op, self.loss, self.perplexity],
                    feed_dict)
                self.dev_summary_writer.add_summary(summaries, step)

            if print_bool:
                print("{}: step {}, loss {:g}, perp {:g}".format(time_str, step, loss, perplexity))

            return (time_str, step, loss, perplexity)
        else:
            last_state_decoder = graph.get_operation_by_name("decoder/last_state_decoder").outputs[0]
            print(session.run([last_state_decoder]))
            #all_vars = tf.trainable_variables()
            #for v in all_vars:
            #    print(v.name)
            #correct_vector = graph.get_operation_by_name("prediction/correct_vector").outputs[0]
            #print(decoder_cell.eval())
            #decoder_cell = session.run([self.decoder_cell])
            #print(decoder_cell)
