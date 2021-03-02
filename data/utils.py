import glob
import math
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
import point_cloud_utils as pcu


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


def test_list(dataset):
    test_files, test_labels = [], []
    for i, obj_type in enumerate(glob.glob(f"{dataset}/*/")):
        cur_files = glob.glob(obj_type + 'test/*.npy')
        test_files.extend(cur_files)
        test_labels.extend([i for _ in range(len(cur_files))])

    return test_files, test_labels


def augment(pc):
    # Add rotation and jitter to point cloud
    theta = np.random.random() * 2 * np.pi
    A = np.array([[np.cos(theta), -np.sin(theta), 0],
                  [np.sin(theta), np.cos(theta), 0],
                  [0, 0, 1]])
    offsets = np.random.normal(0, 0.02, size=pc.shape)
    pt_cloud = np.matmul(pc, A) + offsets
    return pt_cloud


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


def tf_parse_filename_test(filename, label):

    def parse_filename(f):

        # Read in point cloud
        filename_str = f.numpy().decode()
        pt_cloud = np.load(filename_str)
        return pt_cloud

    x = tf.py_function(parse_filename, [filename], [tf.float32])
    x = tf.squeeze(x)
    return x, label


def get_num_classes(dataset):
    if 'ModelNet40' in dataset:
        return 40
    else:
        return 10


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
    moments = tf.concat([second_order_points, cubed_points,
                         tmp_point_moments_1, tmp_point_moments_2, xyz], axis=1)
    return moments, labels


def map_pre_lifting(points, labels):
    sin_pi = tf.sin(math.pi * points)
    cos_pi = tf.cos(math.pi * points)
    sin_2pi = tf.sin(2*math.pi * points)
    cos_2pi = tf.cos(2*math.pi * points)
    harmonics = tf.concat([points, sin_pi, cos_pi, sin_2pi, cos_2pi], axis=1)
    return harmonics, labels


def map_normals(points, labels):

    def estimate_normals(point_cloud):
        normals = pcu.estimate_normals(point_cloud.numpy(), k=50)
        return normals

    first_order_moments = points[:, :3]
    x = tf.py_function(estimate_normals, [first_order_moments], [tf.float32])
    x = tf.squeeze(x)
    points_normals = tf.concat([points, x], axis=1)
    return points_normals, labels
