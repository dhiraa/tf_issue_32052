import argparse
import glob
import os
import shutil
from absl import logging
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from sklearn.datasets import make_regression
import time
from memory_profiler import profile

logging.set_verbosity(logging.INFO)


CRED = '\33[31m'
CGREEN = '\33[32m'
CYELLOW = '\33[33m'
CBLUE = '\33[34m'

CEND = '\33[0m'

def print_info(*args):
    """
    Prints the string in green color
    :param args: user string information
    :return: stdout
    """
    logging.info(CGREEN + str(*args) + CEND)


def print_error(*args):
    """
    Prints the string in red color
    :param args: user string information
    :return: stdout
    """
    logging.error(CRED + str(*args) + CEND)


def print_warn(*args):
    """
    Prints the string in yellow color
    :param args: user string information
    :return: stdout
    """
    logging.warning(CYELLOW + str(*args) + CEND)


def print_debug(*args):
    """
    Prints the string in blue color
    :param args: user string information
    :return: stdout
    """
    logging.debug(CBLUE + str(*args) + CEND)


def memory_usage_psutil(stage_name):
    # return the memory usage in MB
    import psutil
    process = psutil.Process(os.getpid())
    mem = process.memory_info()[0] / float(2 ** 20)
    print_info(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    print_warn(f"{stage_name} : Memory used is {mem}")
    print_info("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")

    return mem


# -----------------------------------------------------------------------------------------------------------------------

# Dataset Handling

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _mat_feature(mat):
    return tf.train.Feature(float_list=tf.train.FloatList(value=mat.flatten()))


def _get_regression_features(data, label):
    """
    Converts numpy array as TF features
    :param data: Numpy Array
    :param label: Numpy Array
    :return:
    """
    return {
        "data": _mat_feature(data),
        "label": _float_feature(label)
    }


def get_numpy_array_size(arr):
    """
    Utility finction to get Numpy Array size
    :param arr: Numpy Array
    :return:
    """
    size = (arr.size * arr.itemsize)/1024/1024
    print_info("%d MBytes " % size)
    return size

@profile
def generate_numpy_tf_records(out_dir,
                              num_tfrecord_files=5,
                              num_samples_per_file=100000,
                              num_features=250):
    """
    Generates random data for Linear Regression and stores them as TFRecords
    :param num_tfrecord_files: Number of TF records
    :param out_dir: Out directory path
    :return:
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for i in tqdm(range(num_tfrecord_files)):
        file_path_name = os.path.join(out_dir, str(i) + ".tfrecords") # ~ 106MB
        print(f"Writing to {file_path_name}")
        if os.path.exists(file_path_name):
            print(f"Found : {file_path_name}")
        else:
            # generate regression dataset
            X, Y = make_regression(n_samples=num_samples_per_file, n_features=num_features, noise=0.1)

            # plot regression dataset

            with tf.io.TFRecordWriter(file_path_name) as writer:
                for x, y in zip(X, Y):
                    # (2500000 * 8) / 1024 /1024 ~ 19MB ~ 10MB on disk
                    # (10000 * 8) / 1024 /1024 ~ 0.076 MB
                    # get_numpy_array_size(features)
                    # get_numpy_array_size(label)
                    #create TF Features
                    features = tf.train.Features(feature=_get_regression_features(data=x, label=y))
                    # create TF Example
                    example = tf.train.Example(features=features)
                    # print(example)
                    writer.write(example.SerializeToString())

@profile
def numpy_array_decode(serialized_example,
                       num_features=250):
    # define a parser
    features = tf.io.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'data': tf.io.FixedLenFeature([1 * num_features], tf.float32),
            'label': tf.io.FixedLenFeature([1], tf.float32),
        })

    data = tf.reshape(
        tf.cast(features['data'], tf.float32), shape=[1, num_features])

    label = tf.reshape(
        tf.cast(features['label'], tf.float32), shape=[1])

    return {"data": data, "dummy": np.random.rand(512, 512, 5)}, label
    # return {"data": data}, label

@profile
def _get_dataset(data_path,
                 batch_size,
                 num_features):
    """
    Reads TFRecords, decode and batches them
    :return: dataset
    """
    _num_cores = 4

    path = os.path.join(data_path, "*.tfrecords")
    path = path.replace("//", "/")
    files = tf.data.Dataset.list_files(path)
    # files = glob.glob(pathname=path)

    # TF dataset APIs
    dataset = files.interleave(
        tf.data.TFRecordDataset,
        cycle_length=_num_cores,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # dataset = tf.data.TFRecordDataset(files, num_parallel_reads=_num_cores)
    # dataset = dataset.shuffle(_batch_size*10, 42)
    # Map the generator output as features as a dict and label

    dataset = dataset.map(map_func=lambda serialized_example : numpy_array_decode(serialized_example=serialized_example,
                                                                                  num_features=num_features),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size=batch_size, drop_remainder=False)
    dataset = dataset.repeat()

    # dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    # iterator = dataset.make_one_shot_iterator()
    # batch_feats, batch_label = iterator.get_next()
    # return batch_feats, batch_label
    return dataset

@profile
def get_tf_records_count(path):
    path = os.path.join(path, "*.tfrecords").replace("//", "/")
    files = glob.glob(path)
    total_records = -1
    for file in tqdm(files, desc="tfrecords size: "):
        # total_records += sum(1 for _ in tf.python_io.tf_record_iterator(file))
        total_records += sum(1 for _ in tf.data.TFRecordDataset(file))
    return total_records

# -----------------------------------------------------------------------------------------------------------------------

@profile
class NNet():
    def __init__(self):
        pass

    def __call__(self, features, labels, params, mode, config=None):
        """
        Used for the :tf_main:`model_fn <estimator/Estimator#__init__>`
        argument when constructing
        :tf_main:`tf.estimator.Estimator <estimator/Estimator>`.
        """
        return self._build(features, labels, params, mode, config=config)

    def _get_optimizer(self, loss):
        with tf.name_scope("optimizer") as scope:

            global_step = tf.compat.v1.train.get_global_step()
            learning_rate = tf.compat.v1.train.exponential_decay(0.001,
                                                                 global_step,
                                                                 decay_steps=100,
                                                                 decay_rate=0.94,
                                                                 staircase=True)

            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate,
                                                 beta_1=0.9,
                                                 beta_2=0.999,
                                                 epsilon=1e-7,
                                                 amsgrad=False,
                                                 name='Adam')

            optimizer.iterations = tf.compat.v1.train.get_or_create_global_step()

            # Get both the unconditional updates (the None part)
            # and the input-conditional updates (the features part).
            # update_ops = model.get_updates_for(None) + model.get_updates_for(features)
            # Compute the minimize_op.
            minimize_op = optimizer.get_updates(
                loss,
                tf.compat.v1.trainable_variables())[0]
            train_op = tf.group(minimize_op)
            return train_op

    def _build(self, features, label, params, mode, config=None):
        memory_usage_psutil("Defining model...")

        features = features['data']

        net = tf.keras.layers.Dense(1024, activation='relu')(features)
        net = tf.keras.layers.Dense(512, activation='relu')(net)
        net = tf.keras.layers.Dense(256, activation='relu')(net)
        net = tf.keras.layers.Dense(128, activation='relu')(net)
        net = tf.keras.layers.Dense(64, activation='relu')(net)
        net = tf.keras.layers.Dense(32, activation='relu')(net)
        logits = tf.keras.layers.Dense(2, activation='softmax')(net)
        classes = tf.math.greater(logits, 0.5)

        loss = None
        optimizer = None
        predictions = {"probability" : logits, "classes" : classes}

        if mode != tf.estimator.ModeKeys.PREDICT:
            mse = tf.keras.losses.MeanSquaredError()
            loss = mse(logits, label)
            tf.summary.scalar('total_loss', loss)

            optimizer = self._get_optimizer(loss=loss)

            tf.summary.scalar('loss', loss)

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            export_outputs={'predict': tf.estimator.export.PredictOutput(predictions)},
            loss=loss,
            train_op=optimizer,
            eval_metric_ops=None)

# -----------------------------------------------------------------------------------------------------------------------
# Estimator Specs
@profile
def _init_tf_config(total_steps_per_file,
                    model_dir,
                    clear_model_data=False,
                    keep_checkpoint_max=5):

    save_checkpoints_steps= total_steps_per_file * 2
    # each TFRecord file has NUM_SAMPLE, so for every 2 TFRecord files store the checkpoint

    save_summary_steps= total_steps_per_file / 5  # log 5 times per file
    log_step_count_steps= total_steps_per_file / 5

    run_config = tf.compat.v1.ConfigProto()
    run_config.gpu_options.allow_growth = True
    # run_config.gpu_options.per_process_gpu_memory_fraction = 0.50
    run_config.allow_soft_placement = True
    run_config.log_device_placement = False
    model_dir = model_dir

    if clear_model_data:
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)

    _run_config = tf.estimator.RunConfig(session_config=run_config,
                                         save_checkpoints_steps=save_checkpoints_steps,
                                         keep_checkpoint_max=keep_checkpoint_max,
                                         save_summary_steps=save_summary_steps,
                                         model_dir=model_dir,
                                         log_step_count_steps=log_step_count_steps)

    return _run_config

@profile
def _get_train_spec(train_data_path, batch_size, num_features, num_epochs=None, max_steps=None):
    # Estimators expect an input_fn to take no arguments.
    # To work around this restriction, we use lambda to capture the arguments and provide the expected interface.
    _total_num_samples = get_tf_records_count(train_data_path)
    steps_per_epoch = _total_num_samples // batch_size

    if max_steps is None:
        max_steps = steps_per_epoch * num_epochs

    return tf.estimator.TrainSpec(
        input_fn=lambda: _get_dataset(data_path=train_data_path,
                                      batch_size=batch_size,
                                      num_features=num_features),
        max_steps=max_steps,
        hooks=None)

@profile
def _get_eval_spec(val_data_path, batch_size, num_features, num_epochs=None, max_steps=None):

    _total_num_samples = get_tf_records_count(val_data_path)
    STEPS_PER_EPOCH = _total_num_samples // batch_size

    if max_steps is None:
        max_steps = STEPS_PER_EPOCH

    return tf.estimator.EvalSpec(
        input_fn=lambda: _get_dataset(data_path=val_data_path,
                                      batch_size=batch_size,
                                      num_features=num_features),
        steps=max_steps,
        hooks=None)


@profile
def train_n_evaluate(estimator,
                     train_data_path,
                     val_data_path,
                     batch_size,
                     num_features,
                     num_epochs=None,
                     max_train_steps=None,
                     max_val_steps=None):
    train_spec = _get_train_spec(train_data_path=train_data_path,
                                 batch_size=batch_size,
                                 num_features=num_features,
                                 num_epochs=num_epochs,
                                 max_steps=max_train_steps)
    eval_spec = _get_eval_spec(val_data_path=val_data_path,
                               batch_size=batch_size,
                               num_features=num_features,
                               num_epochs=num_epochs,
                               max_steps=max_val_steps)

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


def serving_input_receiver_fn(num_features):
    inputs = {
        "data": tf.compat.v1.placeholder(tf.float32, [None, 1, num_features]),
    }
    return tf.estimator.export.ServingInputReceiver(inputs, inputs)


def export_model(estimator, num_features, model_export_path):
    logging.info("Saving model to =======> {}".format(model_export_path))
    if not os.path.exists(model_export_path):
        os.makedirs(model_export_path)
    estimator.export_saved_model(
        model_export_path,
        serving_input_receiver_fn=lambda : serving_input_receiver_fn(num_features=num_features))

# -----------------------------------------------------------------------------------------------------------------------

@profile
def main(args):
    memory_usage_psutil("1. Before generating data")

    #  1. Generate regression data
    generate_numpy_tf_records(out_dir=args["train_path"],
                              num_tfrecord_files=args["num_tfrecord_train_files"],
                              num_samples_per_file=args["num_samples_per_file"],
                              num_features=args["num_features"])
    generate_numpy_tf_records(out_dir=args["val_path"],
                              num_tfrecord_files=2,
                              num_samples_per_file=args["num_samples_per_file"],
                              num_features=args["num_features"])

    total_steps_per_file = args["num_samples_per_file"] / args["batch_size"]

    memory_usage_psutil("2. Before defining model")

    # 2. Define the model
    model = NNet()

    memory_usage_psutil("3. Before defining estimator")

    # 3. Define engine to train i.e Estimator
    estimator = tf.estimator.Estimator(model_fn=model,
                                       config=_init_tf_config(total_steps_per_file=total_steps_per_file,
                                                              model_dir=args["model_dir"]),
                                       params=None)

    memory_usage_psutil("4. Before training")

    # 4. Train and evaluate the model with generated regression data
    train_n_evaluate(estimator=estimator,
                     train_data_path=args["train_path"],
                     val_data_path=args["val_path"],
                     batch_size=args["batch_size"],
                     num_features=args["num_features"],
                     num_epochs=args["num_epochs"],
                     max_train_steps=None,
                     max_val_steps=None)

    memory_usage_psutil("5. Before exporitng the model")

    export_model(estimator=estimator, model_export_path=args["model_export_path"], num_features=args["num_features"])


# generate dataset
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Testing TF Dataset Memory usage : ')

    parser.add_argument('-d', "--delete", type=bool, default=False, help="Delete old data files")
    # parser.add_argument('-m', "--mode", default="", help="[test_iterator]")
    parser.add_argument('-ntf', "--num_tfrecord_train_files", default=5, type=int, help="number of train tfrecord files to generate")
    parser.add_argument('-ntfv', "--num_tfrecord_val_files", default=1,  type=int, help="number of val tfrecord files to generate")
    parser.add_argument('-ns', "--num_samples_per_file", default=10000,  type=int, help="number of samples to generate per file")
    parser.add_argument('-nfeat', "--num_features", default=250,  type=int, help="feature dimension to generate per sample")
    parser.add_argument('-ne', "--num_epochs", default=3,  type=int, help="num of epochs")
    parser.add_argument('-bs', "--batch_size", default=128,  type=int, help="batch size")
    parser.add_argument('-tp', "--train_path", default="data/train_data/", help="path to store train data")
    parser.add_argument('-vp', "--val_path", default="data/val_data/", help="path to store train data")
    parser.add_argument('-mp', "--model_dir", default="data/model/", help="path to store train data")
    parser.add_argument('-mep', "--model_export_path", default="data/model/exported/", help="path to store stripped model data")

    parsed_args = vars(parser.parse_args())

    if parsed_args["delete"]:
        if os.path.exists("data/"):
            shutil.rmtree("data/")

    start_time = time.time()
    main(parsed_args)
    print("--- %s seconds ---" % (time.time() - start_time))

    memory_usage_psutil("Final memory usage: ")

"""
python test_memory_leak.py \
--delete=true \
--num_tfrecord_train_files=5 \
--num_tfrecord_val_files=1 \
--num_samples_per_file=10000 \
--num_features=250 \
--num_epochs=5 \
--batch_size=128 \
--train_path=data/train_data/ \
--val_path=data/val_data/ \
--model_dir=data/model/ \
--model_export_path=data/model/exported/
"""