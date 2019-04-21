"""
Our classification model.
"""
import sys
import pickle
import numpy as np
import tensorflow as tf
#pylint: disable=import-error,no-member,no-name-in-module,not-context-manager
import models.modified_hinge_loss

def invert_dict(dictionary):
    """
    Invert a dict object.
    """
    return {v:k for k, v in dictionary.items()}

def read_local_variables_and_params(arguments):
    """
    Read other variables and params.
    """
    l_vars, _ = pickle.load(
        open(arguments['<data_directory>'] + 'local_variables.pickle', 'rb'))

    #pylint: disable=too-many-instance-attributes,too-few-public-methods
    class Params():
        """
        Parameter class for our model.
        """
        def __init__(self, l_vars, arguments):
            self.output_dim = len(l_vars['ltn'])
            self.pre_trained_embedding_shape = l_vars['word_embedding'].shape
            self.rnn_hidden_neurons = int(arguments['<rnn_hidden_neurons>'])
            self.char_rnn_hidden_neurons = int(arguments['<char_rnn_hidden_neurons>'])
            self.use_mention = arguments['--use_mention']
            self.use_clean = arguments['--use_clean']
            self.embedding_dim = int(arguments['<joint_embedding_size>'])
            self.keep_prob = float(arguments['<keep_prob>'])
            self.learning_rate = float(arguments['<learning_rate>'])
            self.batch_size = int(arguments['<batch_size>'])
            self.char_embedding_shape = (len(l_vars['ctn']) + 1,
                                         int(arguments['<char_embedding_dim>']))
            print(self.char_embedding_shape)
    params = Params(l_vars, arguments)
    return l_vars, params

#pylint: disable=too-many-locals
def read_batch(filenames, batch_size, num_epochs, random=False):
    """
    Read single batch.
    """
    context_features = {
        "lcl": tf.FixedLenFeature([], dtype=tf.int64),
        "rcl": tf.FixedLenFeature([], dtype=tf.int64),
        "eml": tf.FixedLenFeature([], dtype=tf.int64),
        "clean": tf.FixedLenFeature([], dtype=tf.int64),
        "uid": tf.FixedLenFeature([], dtype=tf.string)}
    sequence_features = {
        "labels": tf.FixedLenSequenceFeature([], dtype=tf.int64),
        "lci": tf.FixedLenSequenceFeature([], dtype=tf.int64),
        "rci": tf.FixedLenSequenceFeature([], dtype=tf.int64),
        "emi": tf.FixedLenSequenceFeature([], dtype=tf.int64)}

    with tf.device('/cpu:0'):
        filename_queue = tf.train.string_input_producer(
            filenames,
            num_epochs=num_epochs,
            shuffle=True)

        _, ex = tf.TFRecordReader(
            options=tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)
        ).read(filename_queue)

        if random:
            # maintain a queue of large capacity
            shuffle_queue = tf.RandomShuffleQueue(dtypes=[tf.string],
                                                  capacity=10000,
                                                  min_after_dequeue=1000)
            enqueue_op = shuffle_queue.enqueue(ex)
            dequeue_op = shuffle_queue.dequeue()
            queue_runner = tf.train.QueueRunner(shuffle_queue, [enqueue_op])
        else:
            dequeue_op = ex

        context_parsed, sequence_parsed = tf.parse_single_sequence_example(
            serialized=dequeue_op,
            context_features=context_features,
            sequence_features=sequence_features
            )

        all_examples = {}
        all_examples.update(context_parsed)
        all_examples.update(sequence_parsed)

        if batch_size == 1:
            batch = tf.train.batch(
                tensors=all_examples,
                batch_size=batch_size,
                dynamic_pad=True,
                allow_smaller_final_batch=True,
                name="y_batch"
            )
        else:
            _, batch = tf.contrib.training.bucket_by_sequence_length(
                tf.cast(all_examples['lcl'], tf.int32),
                all_examples,
                batch_size=batch_size,
                bucket_boundaries=[10, 15, 25, 35, 45, 55, 65, 75, 85, 95],
                num_threads=6,
                dynamic_pad=True,
                allow_smaller_final_batch=True
            )
    if random:
        return batch, [queue_runner]
    return batch, []

#pylint: disable=too-many-locals, too-many-statements
def model(batch, parameters, is_training=False):
    """
    Our classification model.
    """
    with tf.name_scope('embeddings'):
        char_embedding = tf.Variable(
            tf.truncated_normal(parameters.char_embedding_shape, stddev=0.05),
            name='char_embeddings',
            trainable=True)

        with tf.device('/cpu:0'):
            word_embedding = tf.Variable(
                tf.constant(0.0, shape=parameters.pre_trained_embedding_shape),
                trainable=False,
                name="word_embedding")

        label_embedding = tf.Variable(
            tf.truncated_normal([parameters.output_dim, parameters.embedding_dim],
                                stddev=0.05),
            name='label_embedding',
            trainable=True)

    keep_prob = tf.Variable(0.0, name='keep_prob', trainable=False)
    with tf.name_scope('fnetClassificationModel'):
        num_neurons = parameters.rnn_hidden_neurons
        cell = tf.contrib.rnn.LSTMCell

        with tf.name_scope('inputs'):
            left_context = tf.nn.embedding_lookup(word_embedding,
                                                  batch['lci'],
                                                  name='left_context')

            mentions = tf.nn.embedding_lookup(char_embedding,
                                              batch['emi'],
                                              name='mention')

            right_context = tf.nn.embedding_lookup(word_embedding,
                                                   batch['rci'],
                                                   name='right_context')

        with tf.name_scope('mention_projections'):
            # mhh if output from last cell
            _, (_, mhh) = tf.nn.dynamic_rnn(cell(parameters.char_rnn_hidden_neurons,
                                                 state_is_tuple=True),
                                            mentions,
                                            batch['eml'],
                                            dtype=tf.float32,
                                            scope='rnn-mentions')
        with tf.name_scope('sentence_projections'):
            # bidirectional encoding of left context
            _, ((_, lcfwh), (_, lcbwh)) = tf.nn.bidirectional_dynamic_rnn(cell(num_neurons,
                                                                               state_is_tuple=True),
                                                                          cell(num_neurons,
                                                                               state_is_tuple=True),
                                                                          left_context,
                                                                          batch['lcl'],
                                                                          dtype=tf.float32,
                                                                          scope='lc-bidirectional')
            # concatenate fwd and bwd pass
            lchh = tf.concat([lcfwh, lcbwh], 1)

            # bidirectional encoding of right context
            _, ((_, rcfwh), (_, rcbwh)) = tf.nn.bidirectional_dynamic_rnn(cell(num_neurons,
                                                                               state_is_tuple=True),
                                                                          cell(num_neurons,
                                                                               state_is_tuple=True),
                                                                          right_context,
                                                                          batch['rcl'],
                                                                          dtype=tf.float32,
                                                                          scope='rc-bidirectional')
            # concatenate fwd and bwd pass
            rchh = tf.concat([rcfwh, rcbwh], 1)
            # return type (outputs, (hidden_states)), (hidden_states) = (State_c, State_h)
            # State_c = cell state, State_h = cell final output.

        with tf.name_scope('output_projection'):
            # combine left and right encoding
            combined = tf.concat([lchh, rchh], 1)
            # apply dropout of left-right combined
            combined_d = tf.nn.dropout(combined, keep_prob, name='dropout_combined')
            # apply dropout on mention representation
            mhhd = tf.nn.dropout(mhh, keep_prob, name='dropout_mention')
            # combine all representations
            if parameters.use_mention:
                all_rep = tf.concat([mhhd, combined_d], 1)
                rep_dim = parameters.char_rnn_hidden_neurons + 4 * parameters.rnn_hidden_neurons
            else:
                all_rep = combined_d
                rep_dim = 4 * parameters.rnn_hidden_neurons

        with tf.name_scope('score-calculation'):
            # a matrix of [input_dim, embedding_space] dim.
            matrix_b = tf.Variable(
                tf.truncated_normal([rep_dim, parameters.embedding_dim],
                                    stddev=0.05),
                name='matrix_b')
            # a matrix of [batch_size, embedding_space] dim.
            transformation = tf.matmul(all_rep, matrix_b)
            # a matrix of shape [batch_size, no_of_labels]
            scores = tf.matmul(transformation, label_embedding, transpose_b=True)

        with tf.name_scope('cost_and_optimization'):
            cost = models.modified_hinge_loss.loss(scores,
                                                   tf.cast(batch['labels'], tf.float32),
                                                   tf.cast(batch['clean'], tf.bool),
                                                   target_dim=parameters.output_dim,
                                                   use_clean=parameters.use_clean)

            train_op = tf.train.AdamOptimizer(parameters.learning_rate).minimize(cost)

        operations = {}
        operations['prediction'] = scores
        operations['cost'] = cost
        operations['optimize'] = train_op
        operations['uid'] = batch['uid']
        if is_training:
            tf.summary.scalar('loss', cost)
            operations['merged_summaries'] = tf.summary.merge_all()
        return operations

def initialize(pre_trained_embedding, keep_prob, session):
    """
    Initizlize model.
    """
    session.run(tf.global_variables_initializer())
    session.run(tf.local_variables_initializer())
    word_embedding_variable = [v for v in tf.global_variables() if v.name == 'embeddings/word_embedding:0'][0]
    keep_prob_var = [v for v in tf.global_variables() if v.name == 'keep_prob:0'][0]
    session.run(word_embedding_variable.assign(pre_trained_embedding))
    session.run(keep_prob_var.assign(keep_prob))

#pylint:disable=too-many-arguments
def train(operations,
          session,
          parameters,
          saver,
          ckpt_directory,
          batches_elapsed,
          save_ckpt_after_sentences=1000000):
    """
    Train model.
    """
    total_cost = []
    save_ckpt_after_batches = int(save_ckpt_after_sentences / parameters.batch_size)
    summary_file_writer = tf.summary.FileWriter(ckpt_directory + '/events', session.graph)
    print('Checkpoint will be saved after ', save_ckpt_after_batches, 'batches')
    while True:
        summary, cost, _ = session.run([operations['merged_summaries'],
                                        operations['cost'],
                                        operations['optimize']])

        batches_elapsed += 1
        summary_file_writer.add_summary(summary, int(batches_elapsed * parameters.batch_size))
        total_cost.append(cost)
        sys.stdout.write('\r{} : mean epoch cost = {}'.format(batches_elapsed,
                                                              np.mean(total_cost)))
        sys.stdout.flush()
        sys.stdout.write('\r')
        if batches_elapsed % save_ckpt_after_batches == 0:
            print('Saving checkpoint: ', save_ckpt_after_batches)
            saver.save(session,
                       ckpt_directory + '/',
                       global_step=int(batches_elapsed * parameters.batch_size),
                       write_meta_graph=False)

    return {
        'cost': np.mean(total_cost)
    }

def test(operations,
         session):
    """
    Test model.
    """
    scores, uids = session.run([operations['prediction'],
                                operations['uid']])

    return scores[0], uids[0]
