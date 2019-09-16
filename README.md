### tf_issue_32052


Code to reproduce the Tensorflow issue @ https://github.com/tensorflow/tensorflow/issues/32052


### How to run ?

```
python test_memory_leak.py \
--delete=true \
--num_tfrecord_train_files=5 \
--num_tfrecord_val_files=1 \
--num_samples_per_file=1000 \
--num_features=250 \
--num_epochs=5 \
--batch_size=128 \
--train_path=data/train_data/ \
--val_path=data/val_data/ \
--model_dir=data/model/ \
--model_export_path=data/model/exported/ \
--use_estimator_train_n_eval=True |& tee log.txt
```

##### [Update 07/Sep/2019] : Old branch https://github.com/dhiraa/tf_issue_32052/tree/master

