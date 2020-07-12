import datetime
import logging
import os

import click
import tensorflow as tf
from tensorflow.keras import models, losses, optimizers, metrics
from tensorflow.keras.layers import Conv2D, MaxPool2D, Input, Conv2DTranspose, UpSampling2D, LeakyReLU, BatchNormalization
from tensorflow.keras.utils import plot_model

from utils.DataGenerator import DataGenerator
from utils.metrics import MAPEavg

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

log = logging.getLogger(__name__)

logging.basicConfig(
    format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s',
    level='INFO'
)


@click.command()
@click.option('--processed-data-dir', type=str, help='Directory with encoded data')
@click.option('--epochs-count', type=int, help='Epochs count for model fit', default=10)
@click.option('--learning-rate', type=float, help='Learning rate for optimizer', default=0.001)
@click.option('--alpha', type=float, help='Parameter for leackyReLU function', default=0.0)
def main(processed_data_dir, epochs_count, learning_rate, alpha):

    log.info('Start')

    log.info('Preparation')
    log_dir = 'logs/fit/' + datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    tf_board_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    train_dir = processed_data_dir + '/train_set'
    val_dir = processed_data_dir + '/val_set'
    test_dir = processed_data_dir + '/test_set'

    train_generator = DataGenerator(
        data_dir=train_dir,
        batch_size=8,
        n_channels=1
    )

    val_generator = DataGenerator(
        data_dir=val_dir,
        batch_size=8,
        shuffle=False,
        n_channels=1
    )

    log.info('Models configure')
    #Encoder
    encoding_input = Input(shape=(512, 512, 1))
    encoding_x = Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same')(encoding_input)
    encoding_x = LeakyReLU(alpha=alpha)(encoding_x)
    encoding_x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(encoding_x)
    encoding_x = BatchNormalization()(encoding_x)
    encoding_x = Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same')(encoding_x)
    encoding_x = LeakyReLU(alpha=alpha)(encoding_x)
    encoding_x = BatchNormalization()(encoding_x)
    encoding_x = Conv2D(filters=1, kernel_size=(3, 3), strides=(2, 2), padding='same')(encoding_x)
    encoding_output = LeakyReLU(alpha=alpha)(encoding_x)


    #Decoder
    decoding_x = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same')(encoding_output)
    decoding_x = LeakyReLU(alpha=alpha)(decoding_x)
    decoding_x = BatchNormalization()(decoding_x)
    decoding_x = UpSampling2D(size=(2, 2))(decoding_x)
    decoding_x = Conv2DTranspose(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same')(decoding_x)
    decoding_x = LeakyReLU(alpha=alpha)(decoding_x)
    decoding_x = UpSampling2D(size=(2, 2))(decoding_x)
    decoding_x = BatchNormalization()(decoding_x)
    decoding_x = Conv2DTranspose(filters=1, kernel_size=(3, 3), strides=(1, 1), padding='same')(decoding_x)
    decoding_output = LeakyReLU(alpha=alpha)(decoding_x)

    #Models
    full_model = models.Model(encoding_input, decoding_output)
    encoder = models.Model(encoding_input, encoding_output)

    plot_model(full_model, show_shapes=True, to_file='Full_model.png')
    plot_model(encoder, show_shapes=True, to_file='Encoder_model.png')

    log.info('Model compiling')
    loss_and_metrics = [
        losses.MeanAbsoluteError(),
        metrics.RootMeanSquaredError(),
        metrics.MeanAbsolutePercentageError(),
        metrics.CosineSimilarity(),
        MAPEavg(batch_shape=[8, 512, 512])
    ]
    full_model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss=loss_and_metrics[0],
        metrics=[loss_and_metrics[1:]]
    )

    print(tf.keras.backend.shape(decoding_output))

    log.info('Model fitting')
    full_model.fit(
        x=train_generator,
        validation_data=val_generator,
        epochs=epochs_count,
        callbacks=[tf_board_callback]
    )

    log.info('Model testing')
    test_generator = DataGenerator(
        data_dir=test_dir,
        batch_size=8,
        shuffle=False,
        n_channels=1
    )

    model_test_results = full_model.evaluate(x=test_generator)
    test_res = {}
    for metric, value in zip(loss_and_metrics, model_test_results):
        test_res.update({metric.name: value})
    log.info(test_res)


if __name__ == '__main__':
    main()
