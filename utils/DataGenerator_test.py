import os
from typing import Tuple, List, NoReturn

import cv2
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras


class DataGenerator(keras.utils.Sequence):
    """
    Data generator for autoencoders
    """
    def __init__(
            self,
            data_dir: str,
            batch_size: int = 1,
            image_size: Tuple[int, int] = (512, 512),
            n_channels: int = 3,
            shuffle: bool = True
    ):
        """
        Data generator for autoencoders
        :param data_dir: Directory with data
        :param batch_size: Batch size
        :param image_size: Size of images
        :param n_channels: Channels count
        :param shuffle:
        """
        self.image_size = image_size
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.data_dir = data_dir
        self.on_epoch_end()

    def __len__(self) -> int:
        """
        Denotes the number of batches per epoch
        :return: Count of steps (batches) per epochs
        """
        return int(len(os.listdir(self.data_dir)) / self.batch_size)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate one batch of data
        :param index: Batch index
        :return:
        """

        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        ids_list = [os.listdir(self.data_dir)[k] for k in indexes]
        batch = self.__data_generation(ids_list)
        return batch, batch

    def on_epoch_end(self) -> NoReturn:
        """
        Updates indexes after each epoch
        :return:
        """
        self.indexes = np.arange(len(os.listdir(self.data_dir)))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, ids_list: List[str]) -> np.ndarray:
        """
        Generate data for batch
        :param ids_list: Ids list
        :return: List with images
        """
        img_list = []
        for img in ids_list:
            img_mat = cv2.imread('/'.join([self.data_dir, img]))
            img_mat = cv2.resize(img_mat, dsize=self.image_size)
            img_list.append(img_mat)
        return np.asarray(img_list)
