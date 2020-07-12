from typing import NoReturn

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import math
from tensorflow.keras.metrics import Metric


class MAPEavg(Metric):

    def __init__(self,  batch_shape: list, name='MAPE_avg'):
        super(MAPEavg, self).__init__(name=name)
        self.avg_mape = tf.Variable(0.0)
        self.shape = batch_shape

    def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight=None) -> NoReturn:
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        y_pred = tf.reshape(y_pred, shape=self.shape)

        mape = math.abs(y_true - y_pred) / K.maximum(y_true, 0.001)
        avg_mape = K.mean(mape)

        self.avg_mape = avg_mape

    def result(self) -> float:
        return self.avg_mape

    def reset_states(self) -> NoReturn:
        self.avg_mape = 0.0