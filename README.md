### tf_issue_32052


Code to reproduce the Tensorflow issue @ https://github.com/tensorflow/tensorflow/issues/32052

`EAST_IMAGE_TEST` flag in `user_config.py` is used to switch between random numpy dataset and image dataset.

Same flag is used to switch between a simple FeedForward regression network and EAST model, 
adopted from https://github.com/argman/EAST

Note: The code in `east_model.py` is still under porting and testing phase from TF 1.x version.
 It may also have something to do with this memory increase.

Functions to look for:
- generate_image_tf_records
- east_features_decode
- _get_dataset


### Observation:
When `EAST_IMAGE_TEST=False`, which used simple FeedForward net and loss, there is no much difference between 
the epochs and the memory usage is constant.
However with EAST Model, which uses different way of optimization routines 
the memory usage spikes with each epoch.


### How to run ?

```
vim user_config.py #modify to your machine setup
python tf_memory_test.py |&  tee log.txt

python parse_objgraph_log.py # works well only when objgraph outputs 50 items :(
xdg-open objgraph_tf_dataset_analysis.csv #each column says how many new objects were added
```