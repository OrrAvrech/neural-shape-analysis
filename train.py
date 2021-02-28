import os
import glob
import random
import logging
import argparse
import importlib
from data.utils import train_val_split, tf_parse_filename, tf_parse_filename_test,\
    get_num_classes, map_second_order_moments, map_third_order_moments, map_pre_lifting, map_normals
import numpy as np
import tensorflow as tf

SEED = 1234
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.compat.v1.set_random_seed(SEED)


def create_datasets(dataset, batch_size, order=1, harmonics=False, normals=False):
    logging.info("Creating train and val datasets...")
    num_classes = get_num_classes(dataset)
    train_files, train_labels, val_files, val_labels = train_val_split(dataset)
    test_files = glob.glob(f"{dataset}/*/test/*.npy")  # only used to get length for comparison
    logging.info(f"Number of training samples: {len(train_files)}")
    logging.info(f"Number of validation samples: {len(val_files)}")
    logging.info(f"Number of testing samples: {len(test_files)}")

    train_ds = tf.data.Dataset.from_tensor_slices((train_files, train_labels))
    val_ds = tf.data.Dataset.from_tensor_slices((val_files, val_labels))
    train_ds = train_ds.shuffle(len(train_files)).map(tf_parse_filename)
    val_ds = val_ds.map(tf_parse_filename_test)
    if order == 1:
        num_channels = 3
    elif order == 2:
        train_ds = train_ds.map(map_second_order_moments)
        val_ds = val_ds.map(map_second_order_moments)
        num_channels = 9
    elif order == 3:
        train_ds = train_ds.map(map_third_order_moments)
        val_ds = val_ds.map(map_third_order_moments)
        num_channels = 19
    else:
        logging.exception('moments order > 3 are not supported')
        raise

    logging.info(f"{order} order moments")
    if harmonics:
        train_ds = train_ds.map(map_pre_lifting)
        val_ds = val_ds.map(map_pre_lifting)
        num_channels += num_channels*4
        logging.info("adding input pre-lifting")
    if normals:
        train_ds = train_ds.map(map_normals)
        val_ds = val_ds.map(map_normals)
        num_channels += 3
        logging.info("adding input point-cloud normals")

    logging.info(f"number of input features: {num_channels}")
    train_ds = train_ds.batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    val_ds = val_ds.batch(batch_size)

    logging.info("Done creating datasets for training")

    return train_ds, val_ds, num_classes, num_channels


def train_model(train_ds, val_ds, num_classes, model, num_channels):
    models_module = importlib.import_module(f"models.{model}")
    model = models_module.get_model(num_classes, num_channels)

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=["sparse_categorical_accuracy"],
    )

    model.fit(train_ds, epochs=3, validation_data=val_ds)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='dataset to use', default='ModelNet40')
    parser.add_argument('--model', help='model to train', default='momenet')
    parser.add_argument('--moment', type=int, default=1, help='moment order [1/2/3]')
    parser.add_argument('--is_normals', help='flag for adding vertex normals', action='store_true')
    parser.add_argument('--is_harmonics', help='flag for adding harmonics pre-lifting', action='store_true')
    parser.add_argument('--num_points', type=int, default=2048, help='Point Number [256/512/1024/2048]')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training')
    return parser.parse_args()


def main():
    args = _parse_args()
    dataset = args.dataset
    batch_size = args.batch_size
    order = args.moment
    is_harmonics = args.is_harmonics
    is_normals = args.is_normals
    model = args.model

    train_ds, val_ds, num_classes, num_channels = create_datasets(dataset, batch_size,
                                                                  order=order,
                                                                  harmonics=is_harmonics,
                                                                  normals=is_normals)
    train_model(train_ds, val_ds, num_classes, model, num_channels)


if __name__ == "__main__":
    main()
