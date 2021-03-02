import argparse
import numpy as np
import matplotlib.pyplot as plt
from data.utils import augment
from mpl_toolkits.mplot3d import Axes3D


def plot(filename, is_augment):
    # Load point cloud
    pt_cloud = np.load(filename)    # N x 3
    if is_augment:
        pt_cloud = augment(pt_cloud)

    # Separate x, y, z coordinates
    xs = pt_cloud[:, 0]
    ys = pt_cloud[:, 1]
    zs = pt_cloud[:, 2]

    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs, ys, zs)
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str, help='File name to point cloud')
    parser.add_argument('--augment', help='point clout augmentation flag', action='store_true')
    args = parser.parse_args()

    filename = args.filename
    is_augment = args.augment
    plot(filename, is_augment)


if __name__ == '__main__':
    main()
