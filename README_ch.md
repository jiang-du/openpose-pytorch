# 基于Pytorch实现的OpenPose人体关节点检测

对发表在期刊T-PAMI 2019的论文"OpenPose: Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields"的pytorch复现。官方的代码在[CMU-Perceptual-Computing-Lab/openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)。

代码完善中，目前仅写完了训练部分，测试的代码还在调试中，预计后续会发布。

For International users, please read to [English version](README.md).

## 环境配置

本项目在Ubuntu 20.04或Windows 10操作系统下都可以运行。作者使用了以下配置环境：

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

## 训练

首先打开配置文件[`config.py`](config.py)根据个人需求修改参数，例如：

```python
stage_define = "PPPPHH"
# 每个stage的定义，P表示PAF场，H表示关节点heatmap
batch_size = 16
num_epochs = 75
learning_rate = 1.0
loader_workers = 8
num_image_pretrain = 8000
print_freq = 20
model_save_filename = './openpose_vgg19.pth'
# 为了方便跨平台共享数据集文件，可以分别对Windows和Linux设置不同的COCO路径
DATA_DIR = './dataset/MSCOCO'
```

然后打开python跑就可以了

```sh
python train.py
```

训练完成后模型会自动保存在[`config.py`](config.py)中设置的那个模型文件名里面。

## 测试

暂时还没有把这部分代码写出来

## 参考文献

> Z. Cao, G. Hidalgo Martinez, T. Simon, S. Wei and Y. A. Sheikh, "OpenPose: Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields," in IEEE Transactions on Pattern Analysis and Machine Intelligence, doi: 10.1109/TPAMI.2019.2929257.

如果这篇论文帮助了你的科研，你可以在你撰写的文章中这样引用latex代码：

```
@article{8765346,
  author = {Z. {Cao} and G. {Hidalgo Martinez} and T. {Simon} and S. {Wei} and Y. A. {Sheikh}},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  title = {OpenPose: Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields},
  year = {2019}
}
```
