import os
import glob
import random
import argparse
import importlib
from data.utils import augment, get_tf_data_sets, train_val_split, tf_parse_filename, get_num_classes, map_second_order_moments
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
    print('Creating train and val datasets...')
    num_classes = get_num_classes(dataset)
    train_files, train_labels, val_files, val_labels = train_val_split(dataset)
    test_files = glob.glob(f"{dataset}/*/test/*.npy")  # only used to get length for comparison
    print('Number of training samples:', len(train_files))
    print('Number of validation samples:', len(val_files))
    print('Number of testing samples:', len(test_files))

    train_ds = tf.data.Dataset.from_tensor_slices((train_files, train_labels))
    train_ds = train_ds.shuffle(len(train_files)).map(tf_parse_filename).map(map_second_order_moments).batch(batch_size)
    train_ds = train_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((val_files, val_labels))
    val_ds = val_ds.map(tf_parse_filename).map(map_second_order_moments).batch(batch_size)
    print('Done!')
    return train_ds, val_ds, num_classes


def train_model(train_ds, val_ds, num_classes, model):
    models_module = importlib.import_module(f"models.{model}")
    model = models_module.get_model(num_classes, 9)

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
    parser.add_argument('--num_points', type=int, default=2048, help='Point Number [256/512/1024/2048]')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training')
    return parser.parse_args()


def main():
    args = _parse_args()
    train_ds, val_ds, num_classes = create_datasets(args.dataset, args.batch_size)
    train_model(train_ds, val_ds, num_classes, args.model)


if __name__ == "__main__":
    main()
