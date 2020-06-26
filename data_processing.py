import logging
import os
import time

import click
from utils.utils import train_val_test_split

log = logging.getLogger(__name__)

logging.basicConfig(
    format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s',
    level='INFO'
)


@click.command()
@click.option('--data-dir', type=str, help='Directory with original data')
@click.option('--processed-data-dir', type=str, help='Directory with encoded data')
@click.option('--train-ratio', type=float, help='Train set ratio', default=0.75)
@click.option('--val-ratio', type=float, help='Validation set ratio', default=0.15)
@click.option('--test-ratio', type=float, help='Test set ratio', default=0.1)
def main(data_dir, processed_data_dir, train_ratio, val_ratio, test_ratio):

    log.info('Start')

    if not os.path.exists(processed_data_dir):
        log.warning('Directory doesn\'t exists')
        os.mkdir(processed_data_dir)
        log.warning('Directory created')

    log.info('Creating directories for validation and test sets')
    train_dir = processed_data_dir + '/train_set'
    val_dir = processed_data_dir + '/' + 'val_set'
    test_dir = processed_data_dir + '/' + 'test_set'
    os.mkdir(train_dir)
    os.mkdir(val_dir)
    os.mkdir(test_dir)

    if train_ratio + val_ratio + test_ratio != 1.0:
        log.error('Wrong ratio sum')
        log.warning('Default ratios will be used')
        train_ratio = 0.75
        val_ratio = 0.15
        test_ratio = 0.1

    log.info(f'Train/Validation/Test split with ratios {train_ratio}/{val_ratio}/{test_ratio} started')
    split_start = time.time()
    train_val_test_split(source_dir=data_dir, train_dir=train_dir, val_dir=val_dir, test_dir=test_dir,
                         train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio)
    split_end = time.time()
    log.info(f'Split end {round(split_end - split_start, 2)}s')
    log.info(f'Train set size: {len(os.listdir(train_dir))}')
    log.info(f'Validation set size: {len(os.listdir(val_dir))}')
    log.info(f'Test set size: {len(os.listdir(test_dir))}')

    log.info('Finish')


if __name__ == '__main__':
    main()
