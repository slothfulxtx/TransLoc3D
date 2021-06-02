# TransLoc3D : Point Cloud based Large-scale Place Recognition using Adaptive Receptive Fields

Paper URL: [https://arxiv.org/abs/2105.11605](https://arxiv.org/abs/2105.11605)

## Abstract

Place recognition plays an essential role in the field of autonomous driving and robot navigation. Although a number of point cloud based methods have been proposed and achieved promising results, few of them take the size difference of objects into consideration. For small objects like pedestrians and vehicles, large receptive fields will capture unrelated information, while small receptive fields would fail to encode complete geometric information for large objects such as buildings. We argue that fixed receptive fields are not well suited for place recognition, and propose a novel Adaptive Receptive Field Module (ARFM), which can adaptively adjust the size of the receptive field based on the input point cloud. We also present a novel network architecture, named TransLoc3D, to obtain discriminative global descriptors of point clouds for the place recognition task. TransLoc3D consists of a 3D sparse convolutional module, an ARFM module, an external transformer network which aims to capture long range dependency and a NetVLAD layer. Experiments show that our method outperforms prior state-of-the-art results, with an improvement of 1.1\% on average recall@1 on the Oxford RobotCar dataset, and 0.8\% on the B.D. dataset.

## Results on Baseline Dataset

| Network          | AR@1     | AR@1\%   |
| ---------------- | -------- | -------- |
| PointNetVLAD     | 63.3     | 80.3     |
| PCAN             | 70.7     | 83.8     |
| DAGC             | 73.3     | 87.5     |
| LPD-Net          | 86.3     | 94.9     |
| SOE-Net          | 89.4     | 96.4     |
| Minkloc3D        | 93.8     | 97.9     |
| NDT-Transformer  | 93.8     | 97.7     |
| Minkloc++        | 93.9     | 98.2     |
| TransLoc3D(ours) | **95.0** | **98.5** |

## Acknowledgments

We would like to sincerely thank [Minkloc3D](https://github.com/jac99/MinkLoc3D), [Minkloc++](https://github.com/jac99/MinkLocMultimodal), [mmdetection](https://github.com/open-mmlab/mmdetection) for their awesome released code.
