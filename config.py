# ----- 网络基本参数设置 -----
# 每个stage的定义，P表示PAF场，H表示关节点heatmap，不能使用其他定义
stage_define = "PPPPHH"
num_stages = len(stage_define)
# 设置batchsize取决于GPU显存，大致上GTX1080对应batchsize=8, Titan RTX对应32左右
batch_size = 48
num_epochs = 150
# ----- 优化器设置 -----
learning_rate = 0.004 # 1.0
weight_decay = 0.0
momentum = 0.9
nesterov = True
# ----- 杂项设置 -----
multi_gpu_train = 1
# disable_continue_train: 打开这个选项将会导致网络从VGG开始train，而不是从上次的结果开始。
# 默认值是True，初次使用之后可以改为False
disable_continue_train = False
# 随机初始化。这项建议不要改，否则很难训练你懂的
train_from_random = False
# 如果GPU使用率太低，可以适当调高一点loader_workers
loader_workers = 8
num_image_pretrain = 8000
print_freq = 20
# ----- 文件相关路径设置 -----
pre_model_name = "pre_model.pth"
model_save_filename = './openpose_vgg19.pth'
import platform
# ----- COCO数据集路径设置 -----
# 为了方便跨平台共享数据集文件，可以分别对Windows和Linux设置不同的COCO路径
if platform.system() == 'Linux':
    DATA_DIR = '/home/ai-lab/code/datasets/coco'
elif platform.system() == 'Windows':
    DATA_DIR = 'H:/dataset/MSCOCO'
else:
    DATA_DIR = '~/MSCOCO'
    raise Exception("Unknown operating system.")

# 生成COCO路径
import os
ANNOTATIONS_TRAIN = [os.path.join(DATA_DIR, 'annotations', item) for item in ['person_keypoints_train2017.json']]
ANNOTATIONS_VAL = os.path.join(DATA_DIR, 'annotations', 'person_keypoints_val2017.json')
IMAGE_DIR_TRAIN = os.path.join(DATA_DIR, 'images/train2017')
IMAGE_DIR_VAL = os.path.join(DATA_DIR, 'images/val2017')

def generate_codec(stage_define):
    stage_codec = list()
    
    for c in stage_define:
        # 强制字符串里面只能使用P和H
        assert ((ord(c) == 80) or (ord(c) == 72))   # 80--P, 72--H
        # 强制类型转换 P--1, H--0
        stage_codec.append((ord(c) - 72) // 8)

    return stage_codec