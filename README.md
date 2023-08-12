# Neural Shape Analysis

This project addresses the task of shape classification of point clouds. 
It is heavily based on PointNet [[1]](#references) and Momenet [[2]](#references), which introduces the usage of geometric moments as input features.
The implementation is in Tensorflow and includes `tf.data.Dataset` mapping functions to compute the geometric moments, pre-liftings and vertex normals.

## Setup
You can create the project's environment from the `nsa.yml` file by running:
```
conda env create -f nsa.yml
```
The point clout vertex normals is based on the Point Cloud Utils ([pcu](https://github.com/fwilliams/point-cloud-utils)) package.

## References
[1] [PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation, C. Qi et al., 2016](https://arxiv.org/pdf/1612.00593.pdf)

[2] [Momen(e)t: Flavor the Moments in Learning to Classify Shapes, M. Joseph-Rivlin et al, 2019](https://arxiv.org/pdf/1812.07431.pdf)
 
