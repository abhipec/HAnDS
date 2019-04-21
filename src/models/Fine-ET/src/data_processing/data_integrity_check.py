"""
Check the integrity of tfrecord files generated.
Display some sample sentences for debugging.
"""

import sys
import pickle
import numpy as np
import tensorflow as tf

def invert_dict(dictionary):
    """
    Invert a dict object.
    """
    return {v:k for k, v in dictionary.items()}

def print_random_mentions_f1(data_directory, filename):
    """
    Print random mentions fron tfrecord file.
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

    tf.reset_default_graph()
    l_vars, _ = pickle.load(open(data_directory + 'local_variables.pickle', 'rb'))

    filename_queue = tf.train.string_input_producer(
        [filename],
        num_epochs=1)

    reader = tf.TFRecordReader(
        options=tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB))


    _, ex = reader.read(filename_queue)

    context_parsed, sequence_parsed = tf.parse_single_sequence_example(
        serialized=ex,
        context_features=context_features,
        sequence_features=sequence_features
        )
    all_examples = {}
    all_examples.update(context_parsed)
    all_examples.update(sequence_parsed)

    batch_size = 100
    batched_data = tf.train.batch(
        tensors=all_examples,
        batch_size=batch_size,
        dynamic_pad=True,
        allow_smaller_final_batch=True,
        name="y_batch"
    )
    sess = tf.Session()

    # Create a coordinator, launch the queue runner threads.
    coord = tf.train.Coordinator()
    sess.run(tf.initialize_all_variables())
    sess.run(tf.initialize_local_variables())
    tf.train.start_queue_runners(sess=sess)
    try:
        while not coord.should_stop():
            # Run training steps or whatever
            out = sess.run(batched_data)
            print_results_f1(out, l_vars, batch_size)
            break
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        # When done, ask the threads to stop.
        coord.request_stop()
    # And wait for them to actually do it.
    sess.close()

def print_results_f1(dictionary, local_variables, batch_size):
    """
    Print results.
    """
    num_to_label = invert_dict(local_variables['ltn'])
    num_to_chrs = invert_dict(local_variables['ctn'])
    num_to_word = invert_dict(local_variables['wtn'])
    for i in range(batch_size):
        print('Uid: ', dictionary['uid'][i])
        mention = ''
        for char_id in dictionary['emi'][i]:
            if num_to_chrs[char_id] == 'eos':
                break
            mention += num_to_chrs[char_id]
        print('Mention: ', mention)
        print('Mention Length: ', dictionary['eml'][i])
        left_context = ''
        for word_id in dictionary['lci'][i]:
            if word_id == 0:
                break
            left_context += num_to_word[word_id] + ' '
        print('Left Context: ', left_context)
        print('Left Context Length: ', dictionary['lcl'][i])
        right_context = ''
        for word_id in dictionary['rci'][i]:
            if word_id == 0:
                break
            right_context += num_to_word[word_id] + ' '
        print('Right Context: ', right_context)
        print('Right Context Length: ', dictionary['rcl'][i])
        labels = [num_to_label[x] for x in np.where(dictionary['labels'][i] > 0)[0]]
        print('Labels: ', labels)
        print('Clean: ', dictionary['clean'][i])
        print('')

if __name__ == '__main__':
    print_random_mentions_f1(sys.argv[1], sys.argv[2])
