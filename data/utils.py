import os
import glob
import math
import trimesh
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
import point_cloud_utils as pcu


def parse_dataset(data_dir, num_points):
    train_points = []
    train_labels = []
    test_points = []
    test_labels = []
    class_map = {}
    folders = glob.glob(os.path.join(data_dir, "[!README]*"))

    for i, folder in enumerate(folders):
        print("processing class: {}".format(os.path.basename(folder)))
        # store folder name with ID so we can retrieve later
        class_map[i] = folder.split("/")[-1]
        # gather all files
        train_files = glob.glob(os.path.join(folder, "train/*"))
        test_files = glob.glob(os.path.join(folder, "test/*"))

        for f in train_files:
            train_points.append(trimesh.load(f).sample(num_points))
            train_labels.append(i)

        for f in test_files:
            test_points.append(trimesh.load(f).sample(num_points))
            test_labels.append(i)

    return (
        np.array(train_points),
        np.array(test_points),
        np.array(train_labels),
        np.array(test_labels),
        class_map,
    )


def train_val_split(dataset, validation_split=0.2):
    train_files, val_files = [], []
    train_labels, val_labels = [], []
    for i, obj_type in enumerate(glob.glob(f"{dataset}/*/")):
        cur_files = glob.glob(obj_type + 'train/*.npy')
        cur_train, cur_val = \
            train_test_split(cur_files, test_size=validation_split)
        train_files.extend(cur_train)
        val_files.extend(cur_val)
        train_labels.extend([i for _ in range(len(cur_train))])
        val_labels.extend([i for _ in range(len(cur_val))])

    return train_files, train_labels, val_files, val_labels


def tf_parse_filename(filename, label):

    def parse_filename(f):

        # Read in point cloud
        filename_str = f.numpy().decode()
        pt_cloud = np.load(filename_str)

        # Add rotation and jitter to point cloud
        theta = np.random.random() * 2*np.pi
        A = np.array([[np.cos(theta), -np.sin(theta), 0],
                      [np.sin(theta), np.cos(theta), 0],
                      [0, 0, 1]])
        offsets = np.random.normal(0, 0.02, size=pt_cloud.shape)
        pt_cloud = np.matmul(pt_cloud, A) + offsets

        return pt_cloud

    x = tf.py_function(parse_filename, [filename], [tf.float32])
    x = tf.squeeze(x)
    return x, label


def get_num_classes(dataset):
    if dataset == 'ModelNet40':
        return 40
    else:
        return 10


def get_tf_data_sets(data_dir, num_points, validation_split=0.2, stratify=True):
    train_points, test_points, train_labels, test_labels, class_map = parse_dataset(data_dir,
                                                                                    num_points)
    if stratify:
        train_points, val_points, train_labels, val_labels = train_test_split(train_points,
                                                                              train_labels,
                                                                              test_size=validation_split,
                                                                              stratify=train_labels)
    else:
        train_points, val_points, train_labels, val_labels = train_test_split(train_points,
                                                                              train_labels,
                                                                              test_size=validation_split)
    train_ds_len = len(train_points)
    train_ds = tf.data.Dataset.from_tensor_slices((train_points, train_labels))
    val_ds = tf.data.Dataset.from_tensor_slices((val_points, val_labels))
    test_ds = tf.data.Dataset.from_tensor_slices((test_points, test_labels))
    train_ds = train_ds.shuffle(train_ds_len)
    return train_ds, val_ds, test_ds


def augment(points, label):
    # jitter points
    points += tf.random.uniform(points.shape, -0.005, 0.005, dtype=tf.float64)
    # shuffle points
    points = tf.random.shuffle(points)
    return points, label


def map_second_order_moments(points, labels):
    # x^2, y^2, z^2
    squared_points = tf.pow(points, 2)
    # xz, yx, zy
    rolled_point_cloud = tf.roll(points, shift=1, axis=-1)
    tmp_point_moments = tf.multiply(points, rolled_point_cloud)
    moments = tf.concat([points, squared_points, tmp_point_moments], axis=1)
    return moments, labels


def map_third_order_moments(points, labels):
    # x^3, y^3, z^3
    cubed_points = tf.pow(points, 3)
    squared_points = tf.pow(points, 2)
    rolled_square = tf.roll(squared_points, shift=1, axis=-1)
    # xxy, yyz, zzx
    tmp_point_moments_1 = tf.multiply(points, rolled_square)
    # xyy, yzz, zxx
    rolled_square = tf.roll(rolled_square, shift=1, axis=-1)
    tmp_point_moments_2 = tf.multiply(points, rolled_square)
    # xyz
    xyz = tf.math.reduce_prod(points, axis=-1, keepdims=True)
    # second order moments
    second_order_points, _ = map_second_order_moments(points, labels)
    moments = tf.concat([points, second_order_points, cubed_points,
                         tmp_point_moments_1, tmp_point_moments_2, xyz])
    return moments, labels


def map_pre_lifting(points, labels):
    sin_pi = tf.sin(math.pi * points)
    cos_pi = tf.cos(math.pi * points)
    sin_2pi = tf.sin(2*math.pi * points)
    cos_2pi = tf.cos(2*math.pi * points)
    harmonics = tf.concat([points, sin_pi, cos_pi, sin_2pi, cos_2pi])
    return harmonics, labels


def map_normals(points, labels):

    def estimate_normals(point_cloud, k=50):
        normals = pcu.estimate_normals(point_cloud, k=k)
        return normals

    x = tf.py_function(estimate_normals, [points], [tf.float32])
    points_normals = tf.concat([points, x])
    return points_normals, labels
