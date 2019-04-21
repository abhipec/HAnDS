"""
Our classification model.
"""
#pylint:disable=not-context-manager
import sys
import pickle
import numpy as np
import tensorflow as tf

def invert_dict(dictionary):
    """
    Invert a dict object.
    """
    return {v:k for k, v in dictionary.items()}

def read_local_variables_and_params(arguments):
    """
    Read other variables and params.
    """
    l_vars, dataset_parameters = pickle.load(
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
            self.keep_prob = float(arguments['<keep_prob>'])
            self.learning_rate = float(arguments['<learning_rate>'])
            self.batch_size = int(arguments['<batch_size>'])
            self.crf_label_string = arguments['<crf_label_string>']
            self.use_char_cnn = arguments['--use_char_cnn']
            self.max_character_sequence_length = int(
                dataset_parameters['<max_character_sequence_length>']
            )
            self.char_embedding_shape = (len(l_vars['ctn']),
                                         int(arguments['<char_embedding_dim>']))
            self.retrain_word_embeddings = arguments['--retrain_word_embeddings']
            self.lstm_layers = int(arguments['<lstm_layers>'])
            self.filter_sizes = [3]
            self.num_filters = 30
    params = Params(l_vars, arguments)
    return l_vars, params


#pylint: disable=too-many-locals,too-many-arguments
def read_batch(filenames,
               batch_size,
               num_epochs,
               label_string,
               use_char_cnn,
               random=False):
    """
    Read single batch.
    """
    context_features = {
        "sl": tf.FixedLenFeature([], dtype=tf.int64)}
    sequence_features = {
        label_string: tf.FixedLenSequenceFeature([], dtype=tf.int64),
        "si": tf.FixedLenSequenceFeature([], dtype=tf.int64)}

    if use_char_cnn:
        sequence_features['ci'] = tf.FixedLenSequenceFeature([], dtype=tf.int64)
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
                                                  capacity=20000,
                                                  min_after_dequeue=2000)
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
                allow_smaller_final_batch=True
            )
        else:

            _, batch = tf.contrib.training.bucket_by_sequence_length(
                tf.cast(all_examples['sl'], tf.int32),
                all_examples,
                batch_size=batch_size,
                bucket_boundaries=[15, 25, 35, 45, 55, 65, 75, 85],
                num_threads=6,
                dynamic_pad=True,
                allow_smaller_final_batch=True
            )
    if random:
        return batch, [queue_runner]
    return batch, []

#pylint: disable=too-many-locals, too-many-statements,invalid-name
def model(batch, parameters, is_training=False):
    """
    Our classification model.
    """
    if parameters.use_char_cnn:
        with tf.name_scope('embeddings'):
            pad_embedding = tf.Variable(tf.zeros([1, parameters.char_embedding_shape[1]]),
                                        name='pad_embedding',
                                        trainable=False)
            char_embedding_w_o_pad = tf.Variable(
                tf.truncated_normal(parameters.char_embedding_shape, stddev=0.01),
                name='char_embeddings_w_o_pad',
                trainable=True)
            char_embedding = tf.concat((pad_embedding, char_embedding_w_o_pad), 0)
    with tf.device('/cpu:0'):
        word_embedding = tf.Variable(
            tf.constant(0.0, shape=parameters.pre_trained_embedding_shape),
            trainable=parameters.retrain_word_embeddings,
            name="word_embedding")
    keep_prob = tf.Variable(0.0, name='keep_prob', trainable=False)

    with tf.name_scope('LinearCRFModel'):
        num_neurons = parameters.rnn_hidden_neurons
        stacked_lstm_fw = tf.contrib.rnn.MultiRNNCell(
            [tf.contrib.rnn.BasicLSTMCell(num_neurons) for _ in range(parameters.lstm_layers)]
        )
        stacked_lstm_bw = tf.contrib.rnn.MultiRNNCell(
            [tf.contrib.rnn.BasicLSTMCell(num_neurons) for _ in range(parameters.lstm_layers)]
        )

        with tf.name_scope('inputs'):
            sentence = tf.nn.embedding_lookup(word_embedding,
                                              batch['si'],
                                              name='sentence')

        with tf.name_scope('sentence_projections'):
            if parameters.use_char_cnn:
                characters = tf.nn.embedding_lookup(char_embedding,
                                                    batch['ci'],
                                                    name='characters')
                batch_size = tf.shape(characters)[0]

                characters_cnn_batch_temp = tf.reshape(characters,
                                                       [batch_size,
                                                        -1,
                                                        parameters.max_character_sequence_length,
                                                        parameters.char_embedding_shape[1]])

                characters_cnn_batch = tf.expand_dims(
                    tf.reshape(characters_cnn_batch_temp,
                               [-1,
                                parameters.max_character_sequence_length,
                                parameters.char_embedding_shape[1]]),
                    -1)
                pooled_outputs = []
                for _, filter_size in enumerate(parameters.filter_sizes):
                    with tf.name_scope("conv-maxpool-%s" % filter_size):
                        filter_shape = [filter_size,
                                        parameters.char_embedding_shape[1],
                                        1,
                                        parameters.num_filters]
                        W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                        b = tf.Variable(tf.constant(0.1, shape=[parameters.num_filters]), name="b")
                        conv = tf.nn.conv2d(
                            characters_cnn_batch,
                            W,
                            strides=[1, 1, 1, 1],
                            padding="VALID",
                            name="conv")
                        # Apply nonlinearity
                        h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                        pooled = tf.nn.max_pool(
                            h,
                            ksize=[1,
                                   parameters.max_character_sequence_length - filter_size + 1,
                                   1,
                                   1],
                            strides=[1, 1, 1, 1],
                            padding='VALID',
                            name="pool")
                        pooled_outputs.append(pooled)
                num_filters_total = parameters.num_filters * len(parameters.filter_sizes)
                h_pool = tf.concat(pooled_outputs, 3)
                h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

                h_pool_sentence_aligned = tf.reshape(h_pool_flat,
                                                     [batch_size, -1, num_filters_total])

                sentence_final = tf.concat([sentence, h_pool_sentence_aligned], 2)
            else:
                sentence_final = sentence
            # bidirectional encoding of left context
            (fw_out, bw_out), _ = tf.nn.bidirectional_dynamic_rnn(stacked_lstm_fw,
                                                                  stacked_lstm_bw,
                                                                  sentence_final,
                                                                  batch['sl'],
                                                                  dtype=tf.float32,
                                                                  scope='sentence-bidirectional')
            # concatenate fwd and bwd pass
            sentence_output = tf.concat([fw_out, bw_out], 2)

            # return type (outputs, (hidden_states)), (hidden_states) = (State_c, State_h)
            # State_c = cell state, State_h = cell final output.

        with tf.name_scope('output_projection'):
            # apply dropout of left-right combined
            sentence_output_d = tf.nn.dropout(sentence_output,
                                              keep_prob,
                                              name='dropout')

        with tf.name_scope('score-calculation'):
            # a matrix of [input_dim, embedding_space] dim.
            projection_matrix = tf.Variable(
                tf.truncated_normal([parameters.rnn_hidden_neurons * 2,
                                     parameters.output_dim], stddev=0.01),
                name='projection_matrix')

            # a matrix of [batch_size, max_seq, vector_size] dim.
            batch_size = tf.shape(sentence_output_d)[0]
            max_seq_length = tf.shape(sentence_output_d)[1]
            vector_size = tf.shape(sentence_output_d)[2]
            transformation = tf.reshape(tf.matmul(tf.reshape(sentence_output_d,
                                                             [-1, vector_size]),
                                                  projection_matrix),
                                        [batch_size, max_seq_length, parameters.output_dim])

        transition_params = tf.Variable(
            tf.constant(1.0/parameters.output_dim,
                        shape=(parameters.output_dim, parameters.output_dim)),
            name="transition_parameters")
        if is_training:
            with tf.name_scope('cost_and_optimization'):
                batch_loss, _ = tf.contrib.crf.crf_log_likelihood(
                    transformation,
                    tf.cast(batch[parameters.crf_label_string], tf.int32),
                    batch['sl'],
                    transition_params
                )
                cost = tf.reduce_mean(-batch_loss)
                # define optimizer
                optimizer = tf.train.AdamOptimizer(learning_rate=parameters.learning_rate)
                # compute gradients
                gvs = optimizer.compute_gradients(cost)
                # clip gradients
                capped_gvs = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gvs]
                train_op = optimizer.apply_gradients(capped_gvs)

        operations = {}
        operations['prediction'] = transformation

        operations['tag_matrix'] = transition_params
        tf.summary.image('CRF matrix', tf.expand_dims(tf.expand_dims(transition_params, 0), 3))

        ## add LSTM weights as summaries
        for i in range(parameters.lstm_layers):
            fw_lstm_cell_weights = [v for v in tf.global_variables()
                                    if v.name == 'sentence-bidirectional/fw/multi_rnn_cell/cell_'
                                    + str(i) + '/basic_lstm_cell/kernel:0'][0]
            bw_lstm_cell_weights = [v for v in tf.global_variables()
                                    if v.name == 'sentence-bidirectional/bw/multi_rnn_cell/cell_'
                                    + str(i) + '/basic_lstm_cell/kernel:0'][0]
            tf.summary.histogram('LSTM_FW_' + str(i), fw_lstm_cell_weights)
            tf.summary.histogram('LSTM_BW_' + str(i), bw_lstm_cell_weights)
        if is_training:
            operations['cost'] = cost
            operations['optimize'] = train_op
            tf.summary.scalar('loss', cost)
        else:
            operations['labels'] = batch[parameters.crf_label_string]
            operations['si'] = batch['si']

        operations['merged_summaries'] = tf.summary.merge_all()
        return operations

def initialize(pre_trained_embedding, keep_prob, session):
    """
    Initizlize model.
    """
    session.run(tf.global_variables_initializer())
    session.run(tf.local_variables_initializer())
    word_embedding_variable = [v for v in tf.global_variables() if v.name == 'word_embedding:0'][0]
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
    scores, tag_matrix, token_indexes, label_indexes = session.run([operations['prediction'],
                                                                    operations['tag_matrix'],
                                                                    operations['si'],
                                                                    operations['labels']])
    prediction, _ = tf.contrib.crf.viterbi_decode(scores[0, :],
                                                  tag_matrix)

    return prediction, token_indexes[0], label_indexes[0]
