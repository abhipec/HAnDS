"""
Main code to run our experiments.
Usage:
    main_crf_test_only <ckpt_directory> <test_file_path>
    main_crf_test_only (-h | --help)

Options:
    -h, --help                      Print this.
    <ckpt_directory>=DIR            Checkpoint directory.
    <test_file_path>=PATH           File path of tfrecord test file.
"""
#pylint:disable=not-context-manager
import os
import glob
import time
import json
import errno
from docopt import docopt
import natsort
import tensorflow as tf
#pylint: disable=no-member
import LCRF as model

def invert_dict(dictionary):
    """
    Invert a dict object.
    """
    return {v:k for k, v in dictionary.items()}

#pylint:disable=raising-bad-type
def ensure_directory(directory):
    """
    Create the directories along the provided directory path that do not exist.
    """
    directory = os.path.expanduser(directory)
    try:
        os.makedirs(directory)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise exception


def load_checkpoint(session,
                    checkpoint_path):
    """
    Load checkpoint if exists.
    """
    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    # filter variables if needed.
    saver_ob = tf.train.Saver(variables, max_to_keep=0)
    saver_ob.restore(session, checkpoint_path[:-6])
    print('loaded model', checkpoint_path)

#pylint: disable=invalid-name
if __name__ == '__main__':
    arguments = docopt(__doc__)
    # load training parameters and update testing parameters
    train_parameters = json.load(open(arguments['<ckpt_directory>'] + 'parameters.json'))
    test_parameters = train_parameters
    test_parameters['<test_file_path>'] = arguments['<test_file_path>']
    test_parameters['<keep_prob>'] = 1
    test_parameters['<batch_size>'] = 1
    test_parameters['<additional_epochs>'] = 1
    test_parameters['<ckpt_directory>'] = arguments['<ckpt_directory>']

    print(test_parameters)

    l_variables, model_parameters = model.read_local_variables_and_params(test_parameters)

    ckpt_directory = test_parameters['<ckpt_directory>']
    checkpoint_files = natsort.natsorted(glob.glob(ckpt_directory + '/*.index'), reverse=True)
    print(checkpoint_files)


    result_directory = ckpt_directory + os.path.basename(test_parameters['<test_file_path>']) + '/'
    ensure_directory(result_directory)

    ntl = invert_dict(l_variables['ltn'])
    ntw = invert_dict(l_variables['wtn'])
    if len(checkpoint_files) > 40:
        print("Too many checkpoints to test.")
        print("Restricting to last 40 checkpoints.")
        print("Because of memory leak.")
    for checkpoint_file in checkpoint_files[:40]:
        with tf.Graph().as_default(), tf.Session() as sess:
            data_batch, _ = model.read_batch(
                [test_parameters['<test_file_path>']],
                test_parameters['<batch_size>'],
                test_parameters['<additional_epochs>'],
                test_parameters['<crf_label_string>'],
                test_parameters['--use_char_cnn'],
                random=False
            )

            ops = model.model(data_batch,
                              model_parameters,
                              is_training=False)

            output_file_path = result_directory + os.path.splitext(
                os.path.basename(checkpoint_file))[0][1:]
            with open(output_file_path, 'w', encoding='utf-8') as file_p:
                predictions, tags, tokens = [], [], []
                # Create a coordinator, launch the queue runner threads.
                coord = tf.train.Coordinator()
                model.initialize(l_variables['word_embedding'], model_parameters.keep_prob, sess)
                tf.train.start_queue_runners(sess=sess)
                time.sleep(1)
                load_checkpoint(sess, checkpoint_file)
                try:
                    while not coord.should_stop():
                        # Run training steps
                        prediction, token_indexes, label_indexes = model.test(ops,
                                                                              sess)
                        for tag in [ntl[x] for x in label_indexes]:
                            tags.append(tag)
                        for tag in [ntl[x] for x in prediction]:
                            predictions.append(tag)
                        for token in [ntw[x] for x in token_indexes]:
                            tokens.append(token)
                        tags.append('\n')
                        tokens.append('\n')
                        predictions.append('\n')
                except tf.errors.OutOfRangeError:
                    print('Done testing -- epoch limit reached')
                except tf.errors.CancelledError:
                    print('Done training -- epoch limit reached counting checkpoints')
                finally:
                    # When done, ask the threads to stop.
                    coord.request_stop()
                sess.close()
                # save output in conll format.
                for token, tag, prediction in zip(tokens, tags, predictions):
                    if token == '\n':
                        row = '\n'
                    else:
                        if tag == 'O':
                            category = tag
                        else:
                            category = tag[2:]
                        row = token + ' ' + category + ' ' + tag + ' ' + prediction + '\n'
                    file_p.write(row)
