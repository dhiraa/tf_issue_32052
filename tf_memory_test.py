import glob
import os
import shutil

import tensorflow as tf
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from absl import logging

# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
from dummy_datasets import _get_dataset, generate_image_tf_records, generate_numpy_tf_records
from east_model import EASTTFModel
from print_helper import memory_usage_psutil, print_error
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

from user_config import *

"""
1. Create TFRecords
2. Define Model as part of Estimator
3. Read TFRecords into Dataset
4. Run Estimator with dataset
5. Collect memory stats
"""

@profile
def _init_tf_config(clear_model_data=False,
                    save_checkpoints_steps=TOTAL_STEPS_PER_FILE * 3,
                    # each TFRecord file has NUM_SAMPLE, so for every 3 TFRecord files store the checkpoint
                    keep_checkpoint_max=5,
                    save_summary_steps=TOTAL_STEPS_PER_FILE * 1,
                    log_step_count_steps=TOTAL_STEPS_PER_FILE * 1):

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
def _get_train_spec(max_steps=None):
    # Estimators expect an input_fn to take no arguments.
    # To work around this restriction, we use lambda to capture the arguments and provide the expected interface.
    return tf.estimator.TrainSpec(
        input_fn=lambda: _get_dataset(data_path=TRAIN_DATA),
        max_steps=max_steps,
        hooks=None)


@profile
def _get_eval_spec(steps):
    return tf.estimator.EvalSpec(
        input_fn=lambda: _get_dataset(data_path=VAL_DATA),
        steps=steps,
        hooks=None)


@profile
def train(estimator, max_steps=None):
    train_spec = _get_train_spec(max_steps=max_steps)
    estimator.train(
        input_fn=train_spec.input_fn,
        hooks=train_spec.hooks,
        max_steps=train_spec.max_steps)


@profile
def evaluate(estimator, steps=None, checkpoint_path=None):
    eval_spec = _get_eval_spec(steps=steps)
    estimator.evaluate(
        input_fn=eval_spec.input_fn,
        steps=eval_spec.steps,
        hooks=eval_spec.hooks,
        checkpoint_path=checkpoint_path)


@profile
def serving_input_receiver_fn():
    if EAST_IMAGE_TEST:
        inputs = {
            "images": tf.compat.v1.placeholder(tf.float32, [None, None, None, 3]),
        }
    else:
        inputs = {
            "data": tf.compat.v1.placeholder(tf.float32, [None, 1, 250]),
        }
    return tf.estimator.export.ServingInputReceiver(inputs, inputs)


@profile
def export_model(estimator, model_export_path):
    logging.info("Saving model to =======> {}".format(model_export_path))
    if not os.path.exists(model_export_path):
        os.makedirs(model_export_path)
    estimator.export_saved_model(
        model_export_path,
        serving_input_receiver_fn=serving_input_receiver_fn)


@profile
def main():
    memory_used = []
    process = psutil.Process(os.getpid())
    if EAST_IMAGE_TEST:
        generate_image_tf_records(number_files=5, out_dir=TRAIN_DATA)
        generate_image_tf_records(number_files=2, out_dir=VAL_DATA)
    else:
        generate_numpy_tf_records(number_files=10, out_dir=TRAIN_DATA)
        generate_numpy_tf_records(number_files=3, out_dir=VAL_DATA)

    # print(dataset_to_iterator(data_path=TRAIN_DATA))

    if EAST_IMAGE_TEST:
        model = EASTTFModel(model_root_directory="store")
    else:
        model = NNet()

    estimator = tf.estimator.Estimator(model_fn=model, config=_init_tf_config(), params=None)
    memory_usage_psutil()
    print('objgraph growth list')
    objgraph.show_growth(limit=50)
    # print(objgraph.get_leaking_objects())

    for epoch in tqdm(range(NUM_EPOCHS)):

        print("\n\n\n\n\n\n")
        print_error(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> New Epoch")
        memory_usage_psutil()
        memory_used.append(process.memory_info()[0] / float(2 ** 20))
        print_error(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Training")
        train(estimator=estimator)
        print_error(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Evaluating")
        evaluate(estimator=estimator)
        print('objgraph growth list after iteration {}'.format(epoch))
        objgraph.show_growth(limit=50)

    plt.plot(memory_used)
    plt.title('Evolution of memory')
    plt.xlabel('iteration')
    plt.ylabel('memory used (MB)')
    plt.savefig("memory_usage.png")
    plt.show()

    print_error(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> New Epoch")
    export_model(estimator=estimator, model_export_path=EXPORT_DIR)

    (objgraph.get_leaking_objects())


if __name__ == '__main__':
    tracemalloc.start()
    main()
    snapshot = tracemalloc.take_snapshot()
    display_top(snapshot)

"""
References:
- https://medium.com/mostly-ai/tensorflow-records-what-they-are-and-how-to-use-them-c46bc4bbb564
"""
