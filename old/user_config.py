import os

#Disable this flag for a normal test run
from print_helper import print_info

IS_EAST_IMAGE_TEST = True

NUM_ARRAYS_PER_FILE = 100000
NUM_FEATURES = 250
NUM_IMAGES_PER_FILE = 8

if IS_EAST_IMAGE_TEST:
  BATCH_SIZE = 4
  TRAIN_DATA  = os.getcwd() + "/data/train_data_img"
  VAL_DATA  = os.getcwd() + "/data/val_data_img"
  MODEL_DIR = os.getcwd() + "/data/" + "east_net"
  EXPORT_DIR = MODEL_DIR + "/" + "export"
  NUM_EPOCHS = 3
  NUM_SAMPLES_PER_FILE = NUM_IMAGES_PER_FILE
else:
  BATCH_SIZE = 128
  TRAIN_DATA  = os.getcwd() + "/data/train_data"
  VAL_DATA  = os.getcwd() + "/data/val_data"
  MODEL_DIR = os.getcwd() + "/" + "data/fwd_nnet"
  EXPORT_DIR = MODEL_DIR + "/" + "export"
  NUM_EPOCHS = 25
  NUM_SAMPLES_PER_FILE = NUM_ARRAYS_PER_FILE

TOTAL_STEPS_PER_FILE = NUM_SAMPLES_PER_FILE / BATCH_SIZE