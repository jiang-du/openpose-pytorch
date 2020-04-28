# ----- 网络基本参数设置 -----
num_stages = 4
batch_size = 8
num_epochs = 75
# ----- 优化器设置 -----
learning_rate = 1.0
weight_decay = 0.0
momentum = 0.9
nesterov = True
# ----- 杂项设置 -----
# disable_continue_train: 打开这个选项将会导致网络从VGG开始train，而不是从上次的结果开始。
# 默认值是Ture，初次使用之后可以改为False
disable_continue_train = False
# 随机初始化。这项建议不要改，否则很难训练你懂的
train_from_random = False
loader_workers = 0
num_image_pretrain = 8000
print_freq = 20
# ----- 文件相关路径设置 -----
pre_model_name = "pre_model.pth"
model_save_filename = './openpose_vgg19.pth'
import platform
# ----- COCO数据集路径设置 -----
# 为了方便跨平台共享数据集文件，可以分别对Windows和Linux设置不同的COCO路径
if platform.system() == 'Linux':
    DATA_DIR = '/media/jiangdu/SOFTWARE/dataset/MSCOCO'
elif platform.system() == 'Windows':
    DATA_DIR = 'H:/dataset/MSCOCO'
else:
    DATA_DIR = '~/MSCOCO'
    raise Exception("Unknown operating system.")

# 生成COCO路径
import os
ANNOTATIONS_TRAIN = [os.path.join(DATA_DIR, 'annotations', item) for item in ['person_keypoints_train2014.json']]
ANNOTATIONS_VAL = os.path.join(DATA_DIR, 'annotations', 'person_keypoints_val2014.json')
IMAGE_DIR_TRAIN = os.path.join(DATA_DIR, 'images/train2014')
IMAGE_DIR_VAL = os.path.join(DATA_DIR, 'images/val2014')

