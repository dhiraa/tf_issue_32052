# from user_config import *
import os

import tensorflow as tf
from tqdm import tqdm
import numpy as np
from sklearn.datasets import make_regression

from memory_profiler import profile


# @profile
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# @profile
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# @profile
def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

# @profile
def _mat_feature(mat):
    return tf.train.Feature(float_list=tf.train.FloatList(value=mat.flatten()))

# @profile
def get_numpy_array_size(arr):
    """
    Utility finction to get Numpy Array size
    :param arr: Numpy Array
    :return:
    """
    size = (arr.size * arr.itemsize)/1024/1024
    print("%d MBytes " % size)
    return size

# @profile
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

# @profile
def _get_east_features(image_mat, score_map_mat, geo_map_mat):
    """
    Given different features matrices, this routine wraps the matrices as TF features
    """
    return {
        "images": _mat_feature(image_mat),
        "score_maps": _mat_feature(score_map_mat),
        "geo_maps": _mat_feature(geo_map_mat),
    }

# @profile
def generate_numpy_tf_records(number_files, 
                              out_dir,
                              NUM_SAMPLES_PER_FILE,
                              NUM_FEATURES):
    """
    Generates random data for Linear Regression and stores them as TFRecords
    :param number_files: Number of TF records
    :param out_dir: Out directory path
    :return:
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for i in tqdm(range(number_files)):
        file_path_name = os.path.join(out_dir, str(i) + ".tfrecords") # ~ 106MB
        print(f"Writing to {file_path_name}")
        if os.path.exists(file_path_name):
            print(f"Found : {file_path_name}")
        else:
            # generate regression dataset
            X, Y = make_regression(n_samples=NUM_SAMPLES_PER_FILE, n_features=NUM_FEATURES, noise=0.1)
            # plot regression dataset

            with tf.io.TFRecordWriter(file_path_name) as writer:
                for x, y in zip(X, Y):
                    # (25000000 * 8) / 1024 /1024 ~ 190MB
                    # (100000 * 8) / 1024 /1024 ~ 0.76 MB
                    # get_numpy_array_size(features)
                    # get_numpy_array_size(label)
                    #create TF Features
                    features = tf.train.Features(feature=_get_regression_features(data=x, label=y))
                    # create TF Example
                    example = tf.train.Example(features=features)
                    # print(example)
                    writer.write(example.SerializeToString())

# @profile
def generate_image_tf_records(number_files, 
                              out_dir,
                              NUM_SAMPLES_PER_FILE):

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for i in tqdm(range(number_files)):
        file_path_name = os.path.join(out_dir, str(i) + ".tfrecords")
        print(f"Writing to {file_path_name}")
        if os.path.exists(file_path_name):
            print(f"Found : {file_path_name}")
            continue
        with tf.io.TFRecordWriter(file_path_name) as writer:
            for i in range(NUM_SAMPLES_PER_FILE):
                image_mat = np.random.rand(512, 512, 3)
                score_map_mat = np.random.rand(128, 128, 1)
                geo_map_mat = np.random.rand(128, 128, 5)
                features = tf.train.Features(
                    feature=_get_east_features(image_mat, score_map_mat, geo_map_mat))
                example = tf.train.Example(features=features)
                writer.write(example.SerializeToString())



@profile
def numpy_array_decode(serialized_example,
                       NUM_FEATURES=250): #TODO make it as arg
    # define a parser
    features = tf.io.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'data': tf.io.FixedLenFeature([1 * NUM_FEATURES], tf.float32),
            'label': tf.io.FixedLenFeature([1], tf.float32),
        })

    data = tf.reshape(
        tf.cast(features['data'], tf.float32), shape=[1, NUM_FEATURES])

    label = tf.reshape(
        tf.cast(features['label'], tf.float32), shape=[1])

    # return {"data": data, "dummy": np.random.rand(512, 512, 5)}, label
    return {"data": data}, label



@profile
def east_features_decode(serialized_example):
    # 1. define a parser
    features = tf.io.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'images': tf.io.FixedLenFeature([512 * 512 * 3], tf.float32),
            'score_maps': tf.io.FixedLenFeature([128 * 128 * 1], tf.float32),
            'geo_maps': tf.io.FixedLenFeature([128 * 128 * 5], tf.float32),
        })

    image = tf.reshape(
        tf.cast(features['images'], tf.float32), shape=[512, 512, 3])
    score_map = tf.reshape(
        tf.cast(features['score_maps'], tf.float32), shape=[128, 128, 1])
    geo_map = tf.reshape(
        tf.cast(features['geo_maps'], tf.float32), shape=[128, 128, 5])

    return {"images": image, "score_maps": score_map, "geo_maps": geo_map}, image #dummy label/Y


@profile
def _get_dataset(data_path,
                 BATCH_SIZE,
                 IS_EAST_IMAGE_TEST):
    """
    Reads TFRecords, decode and batches them
    :return: dataset
    """
    _num_cores = 4
    _batch_size = BATCH_SIZE

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

    if IS_EAST_IMAGE_TEST:
      dataset = dataset.map(map_func=east_features_decode, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    else:
      dataset = dataset.map(map_func=numpy_array_decode, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = dataset.batch(batch_size=_batch_size, drop_remainder=False)
    # dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    iterator = dataset.make_one_shot_iterator()
    batch_feats, batch_label = iterator.get_next()
    return batch_feats, batch_label


@profile
def test_dataset(data_path,
                 BATCH_SIZE,
                 IS_EAST_IMAGE_TEST):
    """
    Reads the TFRecords and creates TF Datasets
    :param data_path:
    :return:
    """
    _num_cores = 4

    path = os.path.join(data_path, "*.tfrecords")
    path = path.replace("//", "/")
    files = tf.data.Dataset.list_files(path)
    # TF dataset APIs
    dataset = files.interleave(
        tf.data.TFRecordDataset,
        cycle_length=_num_cores,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.shuffle(BATCH_SIZE*10, 42)
    # Map the generator output as features as a dict and label
    if IS_EAST_IMAGE_TEST:
        dataset = dataset.map(map_func=east_features_decode, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    else:
        dataset = dataset.map(map_func=numpy_array_decode, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = dataset.batch(batch_size=BATCH_SIZE, drop_remainder=False)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    for features, label in dataset:
        try:
            for key in features.keys():
                print(".", sep="")#print(f"{features[key].shape}", sep= " ")
        #print("\n")
        except:
            print(".", sep="") #hacky way