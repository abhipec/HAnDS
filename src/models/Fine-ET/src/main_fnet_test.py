"""
Main code to run our experiments.
Usage:
    main_crf_test_only <ckpt_directory> <test_file_path> [--hierarchical_prediction]
    main_crf_test_only (-h | --help)

Options:
    -h, --help                      Print this.
    <ckpt_directory>=DIR            Checkpoint directory.
    <test_file_path>=PATH           File path of tfrecord test file.
    [--hierarchical_prediction]     If specified, hierarchical prediction will be used.
"""
#pylint:disable=not-context-manager
import os
import glob
import time
import json
import errno
import natsort
from docopt import docopt
import numpy as np
import tensorflow as tf
#pylint: disable=no-member
#pylint: disable=import-error
import models.FnetClassificationModel as model
from models.evaluation import hierarchical_prediction

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
                random=False
            )

            ops = model.model(data_batch,
                              model_parameters,
                              is_training=False)

            output_file_path = result_directory + os.path.splitext(
                os.path.basename(checkpoint_file))[0][1:]
            with open(output_file_path, 'w', encoding='utf-8') as file_p:
                rows_to_write = []
                sess = tf.Session()
                # Create a coordinator, launch the queue runner threads.
                coord = tf.train.Coordinator()
                model.initialize(l_variables['word_embedding'], model_parameters.keep_prob, sess)
                tf.train.start_queue_runners(sess=sess)
                time.sleep(1)
                load_checkpoint(sess, checkpoint_file)
                try:
                    while not coord.should_stop():
                        # Run training steps
                        scores, uid = model.test(ops,
                                                 sess)
                        if arguments['--hierarchical_prediction']:
                            new_scores = hierarchical_prediction(scores, ntl)
                        else:
                            new_scores = scores > 0
                        labels = [ntl[x] for x in np.nonzero(new_scores)[0]]
                        if not labels:
                            labels = [ntl[np.argmax(scores)]]
                        rows_to_write.append(str(uid, 'utf-8') + '\t' + ','.join(labels))
                except tf.errors.OutOfRangeError:
                    print('Done testing -- epoch limit reached')
                except tf.errors.CancelledError:
                    print('Done training -- epoch limit reached counting checkpoints')
                finally:
                    # When done, ask the threads to stop.
                    coord.request_stop()
                sess.close()
                for row in rows_to_write:
                    file_p.write(row + '\n')
