"""
Check the integrity of tfrecord files generated.
Display some sample sentences for debugging.
"""

import sys
import pickle
import tensorflow as tf

def invert_dict(dictionary):
    """
    Invert a dict object.
    """
    return {v:k for k, v in dictionary.items()}

#pylint:disable=too-many-locals
def print_random_mentions_f1(data_directory, filename):
    """
    Print random mentions fron tfrecord file.
    """
    context_features = {
        "sl": tf.FixedLenFeature([], dtype=tf.int64)}
    sequence_features = {
        "labels": tf.FixedLenSequenceFeature([], dtype=tf.int64),
        "si": tf.FixedLenSequenceFeature([], dtype=tf.int64),
        "ci": tf.FixedLenSequenceFeature([], dtype=tf.int64)}

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

def readable_ems(tokens, tags, length):
    """
    Encapsulate ems withing parenthesis.
    """
    new_tokens = []
    started = False
    for i in range(length):
        if not started:
            if 'B-' in tags[i]:
                new_tokens.append('[[')
                new_tokens.append(tokens[i])
                started = True
            else:
                new_tokens.append(tokens[i])
        elif started:
            if 'I-' in tags[i]:
                new_tokens.append(tokens[i])
            elif tags[i] == 'O':
                started = False
                new_tokens.append(']]')
                new_tokens.append(tokens[i])
            else:
                started = True
                new_tokens.append(']]')
                new_tokens.append('[[')
                new_tokens.append(tokens[i])
    if started:
        new_tokens.append(']]')

    return ' '.join(new_tokens)
def print_results_f1(dictionary, local_variables, batch_size):
    """
    Print results.
    """
    num_to_label = invert_dict(local_variables['ltn'])
    num_to_word = invert_dict(local_variables['wtn'])
    num_to_char = invert_dict(local_variables['ctn'])
    for i in range(batch_size):
        sentence = ''
        tokens = []
        for word_id in dictionary['si'][i]:
            if word_id == 0:
                break
            sentence += num_to_word[word_id] + ' '
            tokens.append(num_to_word[word_id])
        sentence_by_chars = ''
        for char_id in dictionary['ci'][i]:
            if char_id != 0:
                sentence_by_chars += num_to_char[char_id]
                added_space = False
            elif not added_space:
                sentence_by_chars += ' '
                added_space = True
        print('Sentence: ', sentence)
        print('Char Sentence: ', sentence_by_chars)
        print('Sentence Length: ', dictionary['sl'][i])
        tags = [num_to_label[x] for x in dictionary['labels'][i]][:dictionary['sl'][i]]
        print('Entities: ', readable_ems(tokens, tags, dictionary['sl'][i]))
        print(tags)

if __name__ == '__main__':
    print_random_mentions_f1(sys.argv[1], sys.argv[2])
