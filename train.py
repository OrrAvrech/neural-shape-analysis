import os
import glob
import random
import argparse
import importlib
from data.utils import augment, get_tf_data_sets, train_val_split, tf_parse_filename
import numpy as np
import tensorflow as tf

SEED = 1234
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.compat.v1.set_random_seed(SEED)


def create_datasets(batch_size):
    print('Creating train and val datasets...')
    TRAIN_FILES, VAL_FILES = train_val_split()
    TEST_FILES = glob.glob('ModelNet40/*/test/*.npy')  # only used to get length for comparison
    print('Number of training samples:', len(TRAIN_FILES))
    print('Number of validation samples:', len(VAL_FILES))
    print('Number of testing samples:', len(TEST_FILES))
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    train_ds = tf.data.Dataset.list_files(TRAIN_FILES)
    train_ds = train_ds.batch(batch_size, drop_remainder=True)
    train_ds = train_ds.map(tf_parse_filename, num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)

    val_ds = tf.data.Dataset.list_files(VAL_FILES)
    val_ds = val_ds.batch(batch_size, drop_remainder=True)
    val_ds = val_ds.map(tf_parse_filename, num_parallel_calls=AUTOTUNE)
    print('Done!')
    return train_ds, val_ds


def train_model(dataset_dir, num_classes, model, num_points, batch_size):
    train_ds, val_ds, test_ds = get_tf_data_sets(dataset_dir, num_points)
    train_ds = train_ds.map(augment).batch(batch_size)
    val_ds = val_ds.batch(batch_size)
    models_module = importlib.import_module(f"models.{model}")
    model = models_module.get_model(num_points, num_classes, 3)

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=["sparse_categorical_accuracy"],
    )

    model.fit(train_ds, epochs=3, validation_data=val_ds)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir', help='dataset to use')
    parser.add_argument('--num_classes', type=int, help='num of classes in dataset', default=10)
    parser.add_argument('--model', help='model to train', default='momenet')
    parser.add_argument('--num_points', type=int, default=1024, help='Point Number [256/512/1024/2048]')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training')
    return parser.parse_args()


def main():
    args = _parse_args()
    train_model(args.dataset_dir, args.num_classes,
                args.model, args.num_points, args.batch_size)


if __name__ == "__main__":
    main()
