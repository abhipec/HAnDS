"""
Main code to run our experiments.
Usage:
    main_crf <data_directory> <ckpt_dir> <crf_label_string> <search_pattern>
             <rnn_hidden_neurons> <keep_prob> <learning_rate> <batch_size> <char_embedding_dim>
             <lstm_layers> <additional_epochs> <save_ckpt_after_sentences> [--use_char_cnn]
             [--retrain_word_embeddings]
    main_crf (-h | --help)

Options:
    -h, --help                      Print this.
    <data_directory>=<dir>          Data directory.
    <ckpt_dir>=<dir>                Checkpoint directory.
    <crf_label_string>=<name>       Name of label string used for model training.
    <search_pattern>=<string>       This pattern will be used to search for training files.
    <rnn_hidden_neurons>=<number>   Size of RNN hidden units.
    <keep_prob>=<value>             Drop probability value.
    <learning_rate>=<value>         Learning rate.
    <batch_size>=<number>           Batch size to use.
    <char_embedding_dim>=<nimber>   Size of character embedding.
    <lstm_layers>=<number>          Number of LSTM layers.
    <additional_epochs>=<number>    Number of epochs to run.
    <save_ckpt_after_sentences>     Sentence count.
    --use_char_cnn
    --retrain_word_embeddings

"""
import os
import glob
import json
from docopt import docopt
import tensorflow as tf
#pylint: disable=no-member
import LCRF as model

def load_checkpoint(checkpoint_directory,
                    session):
    """
    Load checkpoint if exists.
    """
    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    # filter variables if needed.
    print(variables)
    saver_ob = tf.train.Saver(variables, max_to_keep=0)
    os.makedirs(checkpoint_directory, exist_ok=True)
    # verify if we don't have a checkpoint saved directly
    step = 0
    ckpt = tf.train.get_checkpoint_state(checkpoint_directory)
    if ckpt and ckpt.model_checkpoint_path:
        # Restores from checkpoint
        model_checkpoint_path = ckpt.model_checkpoint_path
        saver_ob.restore(session, model_checkpoint_path)
        step = int(model_checkpoint_path.rsplit('-', 1)[1])
        print('Model loaded = ', step)
    return saver_ob, step

#pylint: disable=invalid-name
if __name__ == '__main__':
    tf.reset_default_graph()
    arguments = docopt(__doc__)
    print(arguments)

    ckpt_directory = arguments['<ckpt_dir>']
    bs = int(arguments['<batch_size>'])

    filenames = list(glob.iglob(arguments['<data_directory>'] + arguments['<search_pattern>']))
    print(filenames)
    data_batch, queue_runners = model.read_batch(filenames,
                                                 bs,
                                                 int(arguments['<additional_epochs>']),
                                                 arguments['<crf_label_string>'],
                                                 arguments['--use_char_cnn'],
                                                 random=True)

    l_variables, parameters = model.read_local_variables_and_params(arguments)
    ops = model.model(data_batch,
                      parameters,
                      is_training=True)

    sess = tf.Session()
    # Create a coordinator, launch the queue runner threads.
    coord = tf.train.Coordinator()
    model.initialize(l_variables['word_embedding'], parameters.keep_prob, sess)
    enqueue_threads = []
    for qr in queue_runners:
        enqueue_threads.append(qr.create_threads(sess, coord=coord, start=True))
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    threads = threads + enqueue_threads

    # Create a saver and session object.
    saver, sentences_elapsed = load_checkpoint(ckpt_directory,
                                               sess)
    batches_elapsed = int(sentences_elapsed / bs)
    # dump parameters used to disk
    with open(ckpt_directory + '/parameters.json', 'w') as json_p:
        json.dump(arguments, json_p, sort_keys=True, indent=4)
    summary_writer = tf.summary.FileWriter(ckpt_directory + '/graph/', sess.graph)

    try:
        while not coord.should_stop():
            # Run training steps
            model.train(ops,
                        sess,
                        parameters,
                        saver,
                        ckpt_directory,
                        batches_elapsed,
                        save_ckpt_after_sentences=int(arguments['<save_ckpt_after_sentences>']))

    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    except tf.errors.CancelledError:
        print('Done training -- epoch limit reached counting checkpoints')
    finally:
        # When done, ask the threads to stop.
        coord.request_stop()
        coord.join(threads)
    sess.close()
