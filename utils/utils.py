import os
import asyncio
from typing import NoReturn

import shutil
import numpy as np


async def async_copyfile(source_name: str, destination_name: str) -> NoReturn:
    """
    Some wrapper function for shutil.copyfile with async statement
    :param source_name: Source file for copying process
    :param destination_name: Destination file for copying process
    """
    shutil.copyfile(source_name, destination_name)


def train_val_test_split(source_dir: str, train_dir: str, val_dir: str, test_dir: str,
                         train_ratio: float, val_ratio: float, test_ratio: float) -> NoReturn:
    """
    Split dataset on 3 parts: train set, validation set and test set.
    Validation and test sets move to its own directories asynchronous.
    :param source_dir: Source directory with data
    :param train_dir: Directory with objects for train
    :param val_dir: Directory with objects for validation
    :param test_dir: Directory with objects for test
    :param train_ratio: Ratio for train set
    :param val_ratio: Ratio for validation set
    :param test_ratio: Ratio for test set
    :return:
    """
    files = os.listdir(source_dir)
    tasks_list = []
    prob_array = np.random.uniform(size=len(files))
    loop = asyncio.get_event_loop()
    for file_name, prob in zip(files, prob_array):
        if prob <= train_ratio:
            source_file = '/'.join((source_dir, file_name))
            dst_dir = '/'.join((train_dir, file_name))
            tasks_list.append(loop.create_task(async_copyfile(source_file, dst_dir)))
        elif train_ratio < prob <= train_ratio + val_ratio:
            source_file = '/'.join((source_dir, file_name))
            dst_dir = '/'.join((val_dir, file_name))
            tasks_list.append(loop.create_task(async_copyfile(source_file, dst_dir)))
        elif train_ratio + val_ratio < prob <= train_ratio + val_ratio + test_ratio:
            source_file = '/'.join((source_dir, file_name))
            dst_dir = '/'.join((test_dir, file_name))
            tasks_list.append(loop.create_task(async_copyfile(source_file, dst_dir)))
    if len(tasks_list) > 0:
        loop.run_until_complete(asyncio.wait(tasks_list))
