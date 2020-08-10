# Pytorch Implementation of OpenPose Body Keypoint Detection

A pytorch implementation of person keypoint detection task in "OpenPose: Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields" at T-PAMI 2019. The official code is at [CMU-Perceptual-Computing-Lab/openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose).

Project in construction. Only training code is available till now. Inference code coming later.

For Chinese users, please read to [Chinese version](README_ch.md).
中文用户请移步[《中文版说明文档》](README_ch.md)。

## Environment

To run this implementation, you need a Ubuntu 20.04 or Windows 10 computer with GPU. The recommended environment is as follows:

```
python >= 3.8.5
cuda == 10.2
cudnn >= 7.6
pytorch >= 1.6
opencv-python >= 4.4
numpy
Pillow
scipy
matplotlib
```

## Train

Firstly modify the settings in [`config.py`](config.py). It depends on your habits. For example:

```python
stage_define = "PPPPHH"
# P for PAF, H for joint heatmap
batch_size = 16
num_epochs = 75
learning_rate = 1.0
loader_workers = 8
num_image_pretrain = 8000
print_freq = 20
model_save_filename = './openpose_vgg19.pth'
# COCO dataset
DATA_DIR = './dataset/MSCOCO'
```

Then simply run:

```sh
python train.py
```

After training, the trained model will be saved according to the filename you set in [`config.py`](config.py).

## Test

I have not published the code for testing yet. Maybe coming soon.

## Reference

> Z. Cao, G. Hidalgo Martinez, T. Simon, S. Wei and Y. A. Sheikh, "OpenPose: Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields," in IEEE Transactions on Pattern Analysis and Machine Intelligence, doi: 10.1109/TPAMI.2019.2929257.

If that paper helps your research, you can cite it using the bibtex:

```
@article{8765346,
  author = {Z. {Cao} and G. {Hidalgo Martinez} and T. {Simon} and S. {Wei} and Y. A. {Sheikh}},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  title = {OpenPose: Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields},
  year = {2019}
}
```
