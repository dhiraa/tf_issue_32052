import tensorflow as tf
from memory_profiler import profile


class NNet(object):
    @profile
    def __init__(self):
        pass

    @profile
    def __call__(self, features, labels, params, mode, config=None):
        """
        Used for the :tf_main:`model_fn <estimator/Estimator#__init__>`
        argument when constructing
        :tf_main:`tf.estimator.Estimator <estimator/Estimator>`.
        """
        return self._build(features, labels, params, mode, config=config)

    @profile
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

    @profile
    def _build(self, features, label, params, mode, config=None):

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

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            export_outputs={'predict': tf.estimator.export.PredictOutput(predictions)},
            loss=loss,
            train_op=optimizer,
            eval_metric_ops=None)