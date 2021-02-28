import os
import glob
import trimesh
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf


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


def train_val_split(train_size=0.8):
    train, val = [], []
    for obj_type in glob.glob('ModelNet40/*/'):
        cur_files = glob.glob(obj_type + 'train/*.npy')
        cur_train, cur_val = \
            train_test_split(cur_files, train_size=train_size, random_state=0, shuffle=True)
        train.extend(cur_train)
        val.extend(cur_val)

    return train, val


def tf_parse_filename(filename):
    """Take batch of filenames and create point cloud and label"""

    idx_lookup = {'airplane': 0, 'bathtub': 1, 'bed': 2, 'bench': 3, 'bookshelf': 4,
                  'bottle': 5, 'bowl': 6, 'car': 7, 'chair': 8, 'cone': 9,
                  'cup': 10, 'curtain': 11, 'desk': 12, 'door': 13, 'dresser': 14,
                  'flower_pot': 15, 'glass_box': 16, 'guitar': 17, 'keyboard': 18,
                  'lamp': 19, 'laptop': 20, 'mantel': 21, 'monitor': 22, 'night_stand': 23,
                  'person': 24, 'piano': 25, 'plant': 26, 'radio': 27, 'range_hood': 28,
                  'sink': 29, 'sofa': 30, 'stairs': 31, 'stool': 32, 'table': 33,
                  'tent': 34, 'toilet': 35, 'tv_stand': 36, 'vase': 37, 'wardrobe': 38,
                  'xbox': 39}

    def parse_filename(filename_batch):

        pt_clouds = []
        labels = []
        for filename in filename_batch:
            # Read in point cloud
            filename_str = filename.numpy().decode()
            pt_cloud = np.load(filename_str)

            # Add rotation and jitter to point cloud
            theta = np.random.random() * 2*3.141
            A = np.array([[np.cos(theta), -np.sin(theta), 0],
                          [np.sin(theta), np.cos(theta), 0],
                          [0, 0, 1]])
            offsets = np.random.normal(0, 0.02, size=pt_cloud.shape)
            pt_cloud = np.matmul(pt_cloud, A) + offsets

            # Create classification label
            obj_type = filename_str.split('/')[1]   # e.g., airplane, bathtub
            label = idx_lookup[obj_type]

            pt_clouds.append(pt_cloud)
            labels.append(label)

        return np.stack(pt_clouds), np.stack(labels)

    x, y = tf.py_function(parse_filename, [filename], [tf.float32, tf.float32])
    return x, y


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
    squared_points = tf.pow(points, 2)
    xy = tf.expand_dims(tf.multiply(points[:, 0], points[:, 1]), axis=1)
    xz = tf.expand_dims(tf.multiply(points[:, 0], points[:, 2]), axis=1)
    yz = tf.expand_dims(tf.multiply(points[:, 1], points[:, 2]), axis=1)
    moments = tf.concat([points, squared_points, xy, xz, yz], axis=1)
    return moments, labels


def map_third_order_moments(points, labels):
    cubed_points = tf.pow(points, 3)
    xxy = tf.expand_dims(tf.multiply(points[:, 0]**2, points[:, 1]), axis=1)
    xxz = tf.expand_dims(tf.multiply(points[:, 0]**2, points[:, 2]), axis=1)
    yyx = tf.expand_dims(tf.multiply(points[:, 1]**2, points[:, 0]), axis=1)
    yyz = tf.expand_dims(tf.multiply(points[:, 1]**2, points[:, 2]), axis=1)
    zzx = tf.expand_dims(tf.multiply(points[:, 2]**2, points[:, 0]), axis=1)
    zzy = tf.expand_dims(tf.multiply(points[:, 2]**2, points[:, 1]), axis=1)
    xyz = tf.expand_dims(points[:, 0]*points[:, 1]*points[:, 2], axis=1)
    second_order_points, _ = map_second_order_moments(points, labels)
    moments = tf.concat([second_order_points, cubed_points, ])
