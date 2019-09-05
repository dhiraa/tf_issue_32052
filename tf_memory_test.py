import glob
import os
import shutil
import argparse
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from absl import logging

# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
from dummy_datasets import _get_dataset, generate_image_tf_records, generate_numpy_tf_records, test_dataset
from east_model import EASTTFModel
from print_helper import memory_usage_psutil, print_error, print_info
from simple_ffwd_net import NNet
from tracemalloc_utils import display_top

logging.set_verbosity(logging.INFO)

#memory profiler
import gc
import psutil
from memory_profiler import profile
import objgraph
import linecache
import os
import tracemalloc

# from user_config import *

"""
1. Create TFRecords
2. Define Model as part of Estimator
3. Read TFRecords into Dataset
4. Run Estimator with dataset
5. Collect memory stats
"""

@profile
def _init_tf_config(TOTAL_STEPS_PER_FILE,
                    MODEL_DIR,
                    clear_model_data=False,
                    keep_checkpoint_max=5):

    save_checkpoints_steps=TOTAL_STEPS_PER_FILE * 3
    # each TFRecord file has NUM_SAMPLE, so for every 3 TFRecord files store the checkpoint

    save_summary_steps=TOTAL_STEPS_PER_FILE * 1
    log_step_count_steps=TOTAL_STEPS_PER_FILE * 1

    run_config = tf.compat.v1.ConfigProto()
    run_config.gpu_options.allow_growth = True
    # run_config.gpu_options.per_process_gpu_memory_fraction = 0.50
    run_config.allow_soft_placement = True
    run_config.log_device_placement = False
    model_dir = MODEL_DIR

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
def _get_train_spec(TRAIN_DATA, BATCH_SIZE, IS_EAST_IMAGE_TEST, max_steps=None):
    # Estimators expect an input_fn to take no arguments.
    # To work around this restriction, we use lambda to capture the arguments and provide the expected interface.
    return tf.estimator.TrainSpec(
        input_fn=lambda: _get_dataset(data_path=TRAIN_DATA,
                                      BATCH_SIZE=BATCH_SIZE,
                                      IS_EAST_IMAGE_TEST=IS_EAST_IMAGE_TEST),
        max_steps=max_steps,
        hooks=None)


@profile
def _get_eval_spec(VAL_DATA, BATCH_SIZE, IS_EAST_IMAGE_TEST, steps):
    return tf.estimator.EvalSpec(
        input_fn=lambda: _get_dataset(data_path=VAL_DATA,
                                      BATCH_SIZE=BATCH_SIZE,
                                      IS_EAST_IMAGE_TEST=IS_EAST_IMAGE_TEST),
        steps=steps,
        hooks=None)


@profile
def train(estimator, TRAIN_DATA, BATCH_SIZE, IS_EAST_IMAGE_TEST, max_steps=None):
    train_spec = _get_train_spec(TRAIN_DATA=TRAIN_DATA,
                                 max_steps=max_steps,
                                 BATCH_SIZE=BATCH_SIZE,
                                 IS_EAST_IMAGE_TEST=IS_EAST_IMAGE_TEST)
    estimator.train(
        input_fn=train_spec.input_fn,
        hooks=train_spec.hooks,
        max_steps=train_spec.max_steps)


@profile
def evaluate(estimator, VAL_DATA, BATCH_SIZE, IS_EAST_IMAGE_TEST, steps=None, checkpoint_path=None):
    eval_spec = _get_eval_spec(VAL_DATA=VAL_DATA, steps=steps,
                               BATCH_SIZE=BATCH_SIZE,
                               IS_EAST_IMAGE_TEST=IS_EAST_IMAGE_TEST)
    estimator.evaluate(
        input_fn=eval_spec.input_fn,
        steps=eval_spec.steps,
        hooks=eval_spec.hooks,
        checkpoint_path=checkpoint_path)


@profile
def serving_input_receiver_fn(IS_EAST_MODEL):
    if IS_EAST_MODEL:
        inputs = {
            "images": tf.compat.v1.placeholder(tf.float32, [None, None, None, 3]),
        }
    else:
        inputs = {
            "data": tf.compat.v1.placeholder(tf.float32, [None, 1, 250]),
        }
    return tf.estimator.export.ServingInputReceiver(inputs, inputs)


@profile
def export_model(estimator, model_export_path, IS_EAST_MODEL):
    logging.info("Saving model to =======> {}".format(model_export_path))
    if not os.path.exists(model_export_path):
        os.makedirs(model_export_path)
    estimator.export_saved_model(
        model_export_path,
        serving_input_receiver_fn=lambda : serving_input_receiver_fn(IS_EAST_MODEL=IS_EAST_MODEL))

@profile
def gen_data(number_files,
             IS_EAST_IMAGE_TEST,
             TRAIN_DATA,
             VAL_DATA,
             NUM_SAMPLES_PER_FILE,
             NUM_FEATURES=None):
    if IS_EAST_IMAGE_TEST:
        generate_image_tf_records(number_files=number_files,
                                  out_dir=TRAIN_DATA,
                                  NUM_SAMPLES_PER_FILE=NUM_SAMPLES_PER_FILE)
        generate_image_tf_records(number_files=1, #TODO Fixed?
                                  out_dir=VAL_DATA,
                                  NUM_SAMPLES_PER_FILE=NUM_SAMPLES_PER_FILE)
    else:
        generate_numpy_tf_records(number_files=number_files,
                                  out_dir=TRAIN_DATA,
                                  NUM_FEATURES=NUM_FEATURES,
                                  NUM_SAMPLES_PER_FILE=NUM_SAMPLES_PER_FILE)
        generate_numpy_tf_records(number_files=2, #TODO fixed?
                                  out_dir=VAL_DATA,
                                  NUM_FEATURES=NUM_FEATURES,
                                  NUM_SAMPLES_PER_FILE=NUM_SAMPLES_PER_FILE)

@profile
def main(args):

    memory_used = []
    process = psutil.Process(os.getpid())

    #TODO add into argparser
    IS_EAST_IMAGE_TEST = True

    NUM_ARRAYS_PER_FILE = 10000

    #TODO decode function needs this value as part of dataset map function,  hence for now harcoded value
    # if needed chnage manually at func `numpy_array_decode` in dummy_dataset.py also
    NUM_FEATURES = 250

    NUM_IMAGES_PER_FILE = 8

    BATCH_SIZE = 4
    TRAIN_DATA = os.getcwd() + "/data/train_data_img"
    VAL_DATA = os.getcwd() + "/data/val_data_img"
    MODEL_DIR = os.getcwd() + "/data/" + "east_net"
    EXPORT_DIR = MODEL_DIR + "/" + "export"
    NUM_EPOCHS = 3
    NUM_SAMPLES_PER_FILE = NUM_IMAGES_PER_FILE


    if args["dataset"] == "numpy":
        IS_EAST_IMAGE_TEST = False
        BATCH_SIZE = 128
        TRAIN_DATA = os.getcwd() + "/data/train_data"
        VAL_DATA = os.getcwd() + "/data/val_data"
        MODEL_DIR = os.getcwd() + "/" + "data/fwd_nnet"
        EXPORT_DIR = MODEL_DIR + "/" + "export"
        NUM_EPOCHS = 5
        NUM_SAMPLES_PER_FILE = NUM_ARRAYS_PER_FILE
    elif args["dataset"] == "east":
        pass
    else:
        print_error("Invalid dataset")

    TOTAL_STEPS_PER_FILE = NUM_SAMPLES_PER_FILE / BATCH_SIZE

    if args["delete"] == True:
        print_info("Deleting old data files")
        shutil.rmtree(TRAIN_DATA)
        shutil.rmtree(VAL_DATA)

    gen_data(IS_EAST_IMAGE_TEST=IS_EAST_IMAGE_TEST,
             TRAIN_DATA=TRAIN_DATA,
             VAL_DATA=VAL_DATA,
             NUM_SAMPLES_PER_FILE=NUM_SAMPLES_PER_FILE,
             NUM_FEATURES=NUM_FEATURES,
             number_files=int(args["num_tfrecord_files"]))

    if args["mode"] == "test_iterator":
        print('objgraph growth list start')
        objgraph.show_growth(limit=50)
        print('objgraph growth list end')


        test_dataset(data_path=TRAIN_DATA,
                     BATCH_SIZE=BATCH_SIZE,
                     IS_EAST_IMAGE_TEST=IS_EAST_IMAGE_TEST)
        test_dataset(data_path=TRAIN_DATA,
                     BATCH_SIZE=BATCH_SIZE,
                     IS_EAST_IMAGE_TEST=IS_EAST_IMAGE_TEST)
        test_dataset(data_path=VAL_DATA,
                     BATCH_SIZE=BATCH_SIZE,
                     IS_EAST_IMAGE_TEST=IS_EAST_IMAGE_TEST)
        print('objgraph growth list start')
        objgraph.show_growth(limit=50)
        print('objgraph growth list end')

        return

    # print(dataset_to_iterator(data_path=TRAIN_DATA))

    if IS_EAST_IMAGE_TEST:
        model = EASTTFModel(model_root_directory="store")
    else:
        model = NNet()

    estimator = tf.estimator.Estimator(model_fn=model,
                                       config=_init_tf_config(TOTAL_STEPS_PER_FILE=TOTAL_STEPS_PER_FILE,
                                                              MODEL_DIR=MODEL_DIR), params=None)
    memory_usage_psutil()
    print('objgraph growth list start')
    objgraph.show_growth(limit=50)
    print('objgraph growth list end')

    # print(objgraph.get_leaking_objects())

    for epoch in tqdm(range(NUM_EPOCHS)):

        print("\n\n\n\n\n\n")
        print_error(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> New Epoch")
        memory_usage_psutil()
        memory_used.append(process.memory_info()[0] / float(2 ** 20))
        print_error(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Training")
        train(estimator=estimator,
              TRAIN_DATA=TRAIN_DATA,
              BATCH_SIZE=BATCH_SIZE,
              IS_EAST_IMAGE_TEST=IS_EAST_IMAGE_TEST)
        print_error(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Evaluating")
        evaluate(estimator=estimator,
                 VAL_DATA=VAL_DATA,
                 BATCH_SIZE=BATCH_SIZE,
                 IS_EAST_IMAGE_TEST=IS_EAST_IMAGE_TEST)
        print('objgraph growth list start')
        objgraph.show_growth(limit=50)
        print('objgraph growth list end')


    plt.plot(memory_used)
    plt.title('Evolution of memory')
    plt.xlabel('iteration')
    plt.ylabel('memory used (MB)')
    plt.savefig("logs/" + args["dataset"] + "_dataset_memory_usage.png")
    plt.show()

    print_error(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> New Epoch")
    export_model(estimator=estimator, model_export_path=EXPORT_DIR, IS_EAST_MODEL=IS_EAST_IMAGE_TEST)

    (objgraph.get_leaking_objects())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Testing TF Dataset Memory usage : ')

    parser.add_argument('-d', "--delete", type=bool, default=False, help="Delete old data files")
    parser.add_argument('-m', "--mode", default="", help="[test_iterator]")
    parser.add_argument('-ds', "--dataset", default="east", help="[east/numpy]")
    parser.add_argument('-nf', "--num_tfrecord_files", default=5, help="number of train tfrecord files to generate")


    parsed_args = vars(parser.parse_args())

    print_error(parsed_args)
    tracemalloc.start()
    main(parsed_args)
    snapshot = tracemalloc.take_snapshot()
    display_top(snapshot)

"""
References:
- https://medium.com/mostly-ai/tensorflow-records-what-they-are-and-how-to-use-them-c46bc4bbb564
"""
