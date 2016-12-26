import tensorflow as tf
import numpy as np
from tensorflow.contrib import learn
import datetime

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.curdir)[:-7])

import tf_helpers
import data_utils
from RNNClassification import RNNClassifier

# Code based on: https://github.com/dennybritz/cnn-text-classification-tf

embedding_dim = 100
batch_size = 54
hidden_size = 100
num_classes = 5
num_epochs = 10
learning_rate = 0.001

# Load Data
# =================================================================================
train_filename = "pnp-train.txt"
validate_filename = "pnp-validate.txt"
test_filename = "pnp-test.txt"

x_train, y_train, seq_lens_train, vocab_dict \
    = data_utils.load_data_and_labels(train_filename)
train_data = list(zip(x_train, y_train, seq_lens_train))

x_dev, y_dev, seq_lens_dev, vocab_dict \
    = data_utils.load_data_and_labels(validate_filename, train=False)
dev_data = list(zip(x_dev, y_dev, seq_lens_dev))

# Helper Functions
# =================================================================================
def train_step(x_batch, y_batch, seq_lens, current_step, writer=None, print_bool=False):
    """
    Single training step
    """
    feed_dict = {
        nn.input_x: x_batch,
        nn.input_y: y_batch,
        nn.seq_lens: seq_lens
    }
    _, summaries, loss_val, accuracy_val = sess.run([train_op, train_summary_op, nn.loss, nn.accuracy], feed_dict)

    time_str = datetime.datetime.now().isoformat()
    if print_bool:
        print("\nTrain: {}: step {}, loss {:g}, acc {:g}".format(time_str, current_step, loss_val, accuracy_val))
    if writer:
        writer.add_summary(summaries, current_step)

    return (loss_val, accuracy_val)

def dev_eval(dev_data, current_step, writer=None):
    """
    Evaluates model on a validation set
    """
    batches = data_utils.batch_iter(dev_data, batch_size=batch_size, num_epochs=1, shuffle=False)
    loss_val = 0
    accuracy_val = 0

    batch_count = 0
    for dev_batch in batches:
        batch_count += 1

        x_batch = dev_batch[:,0]
        y_batch = dev_batch[:,1]
        seq_lens_batch = dev_batch[:,2]

        x_batch = data_utils.pad(x_batch, seq_lens_batch)

        loss_val_batch, accuracy_val_batch = dev_step(x_batch, y_batch, seq_lens_batch, \
            current_step, writer=writer)
        loss_val += loss_val_batch
        accuracy_val += accuracy_val_batch

    time_str = datetime.datetime.now().isoformat()
    print("Dev:   {}: step {}, loss {:g}, acc {:g}".format(time_str, current_step, \
        loss_val/batch_count, accuracy_val/batch_count))

    return (loss_val, accuracy_val)

def dev_step(x_batch, y_batch, seq_lens, current_step, writer=None):
    """
    Evaluates model on a validation set
    """

    x_batch_padded = data_utils.pad(x_batch, seq_lens)

    feed_dict = {
        nn.input_x: x_batch_padded,
        nn.input_y: y_batch,
        nn.seq_lens: seq_lens
    }
    summaries, loss_val, accuracy_val = sess.run([dev_summary_op, nn.loss, nn.accuracy], feed_dict)
    if writer:
        writer.add_summary(summaries, current_step)

    return (loss_val, accuracy_val)


# Starting Session
# ================================================================================

graph = tf.Graph()
sess = tf.InteractiveSession()
nn = RNNClassifier(
        num_classes=num_classes,
        vocab_size=len(vocab_dict),
        hidden_size=hidden_size,
        embedding_dim=embedding_dim,
        batch_size=batch_size,
        bidirectional=False
    )

optimizer = tf.train.AdamOptimizer(learning_rate) # TODO: CHOOSE YOUR FAVORITE OPTIMZER
global_step = tf.Variable(0, name='global_step', trainable=False)
grads_and_vars = optimizer.compute_gradients(nn.loss)
train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

train_summary_op, dev_summary_op, train_summary_writer, dev_summary_writer, timestamp, checkpoint_prefix = \
    tf_helpers.save_summaries(sess, nn.loss, nn.accuracy, grads_and_vars)
saver = tf.train.Saver(tf.all_variables())

# Training and Validation
# ===============================================================================
sess.run(tf.initialize_all_variables())

def loss_early_stopping():
    min_loss = 999999999
    increasing_loss_count = 0
    max_accuracy = 0
    max_accuracy_step = 0

    for batch in batches:
        x_batch, y_batch = zip(*batch) # TODO: SETUP YOUR DATA'S BATCHES

        current_step = tf.train.global_step(sess, global_step)
        if current_step % 500 == 0:
            train_loss, train_accuracy = train_step(x_batch, y_batch, current_step, print_bool=True)
            dev_loss, dev_accuracy = dev_step(x_dev, y_dev, current_step)

            if dev_loss < min_loss:
                min_loss = dev_loss
                increasing_loss_count = 0
            else:
                increasing_loss_count += 1

            if dev_accuracy > max_accuracy:
                max_accuracy = dev_accuracy
                max_accuracy_step = current_step

            if current_step > FLAGS.patience and FLAGS.patience_increase < increasing_loss_count:
                break

        else:
            train_loss, train_accuracy = train_step(x_batch, y_batch, current_step, print_bool=False)

        if current_step % FLAGS.checkpoint_every == 0:
            path = saver.save(sess, checkpoint_prefix, global_step=global_step)
            print("Saved model checkpoint to {}".format(path))

    return (train_loss, train_accuracy, max_accuracy, max_accuracy_step)

def accuracy_early_stopping():
    max_accuracy = 0
    max_accuracy_step = 0

    #for batch in batches:
        #x_batch, y_batch = zip(*batch) # TODO: SETUP YOUR DATA'S BATCHES

    for _ in range(1000):
        x_batch, y_batch = mnist.train.next_batch(FLAGS.batch_size)

        current_step = tf.train.global_step(sess, global_step)
        if current_step % FLAGS.evaluate_every == 0:
            train_loss, train_accuracy = train_step(x_batch, y_batch, current_step, print_bool=True)
            dev_loss, dev_accuracy = dev_step(x_dev, y_dev, current_step)

            if dev_accuracy > max_accuracy:
                max_accuracy = dev_accuracy
                max_accuracy_step = current_step

            if current_step > FLAGS.patience and FLAGS.patience_increase < current_step - max_accuracy_step:
                break

        else:
            train_loss, train_accuracy = train_step(x_batch, y_batch, current_step, print_bool=False)

        if current_step % FLAGS.checkpoint_every == 0:
            path = saver.save(sess, checkpoint_prefix, global_step=global_step)
            print("Saved model checkpoint to {}".format(path))

    return (train_loss, train_accuracy, max_accuracy, max_accuracy_step)

def run_for_epochs(batches):
    for batch in batches:
        x_batch = batch[:,0]
        y_batch = batch[:,1]
        seq_lens_batch = batch[:,2]

        x_batch = data_utils.pad(x_batch, seq_lens_batch)

        current_step = tf.train.global_step(sess, global_step)
        if current_step % 100 == 0:
            train_loss, train_accuracy = train_step(x_batch, y_batch, seq_lens_batch, \
                current_step, print_bool=True)
            dev_loss, dev_accuracy = dev_eval(dev_data, current_step, writer=dev_summary_writer)

        else:
            train_loss, train_accuracy = train_step(x_batch, y_batch, seq_lens_batch, \
                current_step, print_bool=False)

        if current_step % 500 == 0:
            path = saver.save(sess, checkpoint_prefix, global_step=global_step)
            print("Saved model checkpoint to {}".format(path))

    return (train_loss, train_accuracy)

batches = data_utils.batch_iter(train_data, batch_size=batch_size, num_epochs=num_epochs, shuffle=True)
train_loss, train_accuracy = run_for_epochs(batches)

print("\nFinal Valildation Evaluation:")
current_step = tf.train.global_step(sess, global_step)
dev_loss, dev_accuracy = dev_eval(dev_data, current_step, writer=dev_summary_writer)
#print("Maximum validation accuracy at step {}: {}".format(max_accuracy_step, max_accuracy))
print("")

tf_helpers.write_results(current_step, train_loss, train_accuracy, dev_loss, dev_accuracy, timestamp)

"""
for v in tf.all_variables():
    print(v.name)

input_x = graph.as_graph_element("input_x:0").outputs[0]
input_y = graph.as_graph_element("input_y:0").outputs[0]
seq_lens = graph.as_graph_element("seq_lens:0").outputs[0]

predictions = graph.get_operation_by_name("output/predictions").outputs[0]
num_correct = graph.get_operation_by_name("output/num_correct").outputs[0]
"""

sess.close()
