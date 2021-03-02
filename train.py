import os
import json
import random
import zipfile
import logging
import argparse
import importlib
from data.utils import train_val_split, test_list, tf_parse_filename, tf_parse_filename_test,\
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


def find_in_path(path, suffix):
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in [f for f in filenames if f.endswith(suffix)]:
            return os.path.join(dirpath, filename)


def unzip_file(zip_file, extract_dir):
    logging.info(f"Extract {zip_file} to {extract_dir}")
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)


def create_datasets(dataset, batch_size, order=1, harmonics=False, normals=False):
    dataset_path = find_in_path(dataset, ".zip")
    logging.debug(f"dataset-{dataset}_path-{dataset_path}")
    if dataset_path:
        dataset_dir = os.path.dirname(dataset_path)
        dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
        logging.debug(f"ds_name-{dataset_name}")
        unzip_file(dataset_path, dataset_dir)
        dataset = os.path.join(dataset_dir, dataset_name)
    logging.info("Creating train, validation and test datasets...")
    num_classes = get_num_classes(dataset)
    train_files, train_labels, val_files, val_labels = train_val_split(dataset)
    test_files, test_labels = test_list(dataset)
    logging.info(f"Number of training samples: {len(train_files)}")
    logging.info(f"Number of validation samples: {len(val_files)}")
    logging.info(f"Number of testing samples: {len(test_files)}")

    train_ds = tf.data.Dataset.from_tensor_slices((train_files, train_labels))
    val_ds = tf.data.Dataset.from_tensor_slices((val_files, val_labels))
    test_ds = tf.data.Dataset.from_tensor_slices((test_files, test_labels))
    train_ds = train_ds.shuffle(len(train_files)).map(tf_parse_filename)
    val_ds = val_ds.map(tf_parse_filename_test)
    test_ds = test_ds.map(tf_parse_filename_test)
    if order == 1:
        num_channels = 3
    elif order == 2:
        train_ds = train_ds.map(map_second_order_moments)
        val_ds = val_ds.map(map_second_order_moments)
        test_ds = test_ds.map(map_second_order_moments)
        num_channels = 9
    elif order == 3:
        train_ds = train_ds.map(map_third_order_moments)
        val_ds = val_ds.map(map_third_order_moments)
        test_ds = test_ds.map(map_third_order_moments)
        num_channels = 19
    else:
        logging.exception('moments order > 3 are not supported')
        raise

    logging.info(f"{order} order moments")
    if harmonics:
        train_ds = train_ds.map(map_pre_lifting)
        val_ds = val_ds.map(map_pre_lifting)
        test_ds = test_ds.map(map_pre_lifting)
        num_channels += num_channels*4
        logging.info("adding input pre-lifting")
    if normals:
        train_ds = train_ds.map(map_normals)
        val_ds = val_ds.map(map_normals)
        test_ds = test_ds.map(map_normals)
        num_channels += 3
        logging.info("adding input point-cloud normals")

    logging.info(f"number of input features: {num_channels}")
    train_ds = train_ds.batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    val_ds = val_ds.batch(batch_size)
    test_ds = test_ds.batch(batch_size)

    logging.info("Done creating datasets for training")

    return train_ds, val_ds, test_ds, num_classes, num_channels


def train_model(train_ds, val_ds, num_classes, model, num_channels, lr, logs_dir, export_dir, epochs):
    models_module = importlib.import_module(f"models.{model}")
    model = models_module.get_model(num_classes, num_channels)

    # Callbacks #
    # Saved model
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(export_dir)
    # Early stopping
    early_stop_cb = tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True)
    # Tensorboard
    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=logs_dir)
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        metrics=["sparse_categorical_accuracy"],
    )

    model.fit(train_ds, epochs=epochs, validation_data=val_ds, callbacks=[checkpoint_cb,
                                                                          early_stop_cb,
                                                                          tensorboard_cb])
    return model


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='dataset directory or zip file', default='ModelNet40')
    parser.add_argument('--model', help='model to train', default='momenet')
    parser.add_argument('--moment', type=int, default=1, help='moment order [1/2/3]')
    parser.add_argument('--is_normals', type=int, default=0, help='flag for adding vertex normals')
    parser.add_argument('--is_harmonics', type=int, default=0, help='flag for adding harmonics pre-lifting')
    parser.add_argument('--num_points', type=int, default=2048, help='Point Number [256/512/1024/2048]')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning-rate hyper-parameter')
    parser.add_argument('--logs_dir', type=str, help='tensorboard logs dir')
    parser.add_argument('--export_dir', type=str, help='export model dir')
    parser.add_argument('--metrics_path', type=str, help='output file for recording metrics')
    parser.add_argument('--epochs', type=int, default=15, help='number of training epochs')
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.DEBUG)

    args = _parse_args()
    dataset = args.dataset
    batch_size = args.batch_size
    order = args.moment
    is_harmonics = args.is_harmonics
    is_normals = args.is_normals
    model = args.model
    lr = args.learning_rate
    epochs = args.epochs
    logs_dir = args.logs_dir
    export_dir = args.export_dir
    metrics_path = args.metrics_path
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(export_dir, exist_ok=True)

    train_ds, val_ds, test_ds, num_classes, num_channels = create_datasets(dataset, batch_size,
                                                                           order=order,
                                                                           harmonics=is_harmonics,
                                                                           normals=is_normals)
    trained_model = train_model(train_ds, val_ds,
                                num_classes, model, num_channels,
                                lr, logs_dir, export_dir, epochs)
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    test_results = trained_model.evaluate(test_ds, return_dict=True)
    with open(metrics_path, 'w') as metrics_file:
        json.dump(test_results, metrics_file)


if __name__ == "__main__":
    main()
