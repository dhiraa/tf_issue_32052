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


### How to run ?

```
python tf_memory_test.py --mode=test_east_iterator --dataset=east |&  tee east_itr_log.txt
python tf_memory_test.py --mode=test_east_iterator --dataset=numpy |&  tee numpy_itr_log.txt

```
- Dataset APIs consumes memory by loading the TFRecord files

```
python tf_memory_test.py --dataset=numpy |&  tee simpe_net_log.txt
```

-  When `mode` arg is set to `simple_net`, which uses simple FeedForward net and loss, there is no much difference between 
the epochs and the memory usage is in sub-linear increase. #TODO retest this!

```
python tf_memory_test.py --dataset=east |&  tee east_log.txt
```

- However with EAST Model, which uses different way of optimization routines 
the memory usage spikes with each epoch.


#### TODO
```python parse_objgraph_log.py # works well only when objgraph outputs 50 items :(
xdg-open objgraph_tf_dataset_analysis.csv #each column says how many new objects were added
```

