### tf_issue_32052


Code to reproduce the Tensorflow issue @ https://github.com/tensorflow/tensorflow/issues/32052

`mode` arg is used to switch between random numpy dataset and image dataset.

Same arg is used to switch between a simple FeedForward regression network and EAST model, 
adopted from https://github.com/argman/EAST

Note: The code in `east_model.py` is still under porting and testing phase from TF 1.x version.
 It may also have something to do with this memory increase.

Functions to look for:
- [generate_image_tf_records](dummy_datasets.py)
- [ east_features_decode](dummy_datasets.py)
- [_get_dataset](dummy_datasets.py)

### Configurations

All UPPERCASE VARIABLES are configurations in [tf_memory_test.py](tf_memory_test.py)

**Numpy TFRecords**

```python
NUM_ARRAYS_PER_FILE = 10000
NUM_FEATURES = 250
```

Size = 10000 * 1 * 250 = 2500000 * 8bytes / 1024 / 1024 ~ 20MB ~ 10MB on disk

**EAST Dummy Image TFRecords**

```python
                ...
                image_mat = np.random.rand(512, 512, 3)
                score_map_mat = np.random.rand(128, 128, 1)
                geo_map_mat = np.random.rand(128, 128, 5)
                ...
                
NUM_IMAGES_PER_FILE = 8
```

(((512 * 512 * 3) + (128 * 128 * 1) + (128 * 128 * 5) ) * 8 bytes) * 8 files / 1024 / 1024 = 54MB ~ 28MB on disk

### How to run ?

**Test Iterator**

```
python tf_memory_test.py --delete=true  --num_tfrecord_files=3 --mode=test_iterator --dataset=east |&  tee logs/east_itr_log.txt
python tf_memory_test.py --delete=true  --num_tfrecord_files=6  --mode=test_iterator --dataset=numpy |&  tee logs/numpy_itr_log.txt

```
- Dataset APIs consumes memory by loading the TFRecord files


**Test Estimators**
```
python tf_memory_test.py --dataset=numpy |&  tee logs/simpe_net_log.txt
python tf_memory_test.py --delete=true  --num_tfrecord_files=3 --dataset=east |&  tee logs/east_model_log.txt
```


#### Objgraph information parser

```python parse_objgraph_log.py 
python parse_objgraph_log.py -f=east_itr_log.txt
cat log.txt #for pretty colorful prints
```

