import torch

def load_weights(model):
    import os
    from config import disable_continue_train, model_save_filename, pre_model_name, train_from_random
    if not(disable_continue_train):
        # 判断是否存在模型文件
        if os.path.isfile(model_save_filename):
            # 加载当前最佳模型
            model.load_state_dict(torch.load(model_save_filename))
            return 1
        elif os.path.isfile(pre_model_name):
            # 加载5个epoch之后的模型
            model.load_state_dict(torch.load(pre_model_name))
            return 2
        else:
            # 没有weight文件，加载个锤子啊
            raise Exception("No weight file. Would you like to train with your hammer?")
    else:
        if train_from_random:
            # 随机初始值，不加载任何权重
            return 0
        else:
            # 加载Imagenet VGG19权重
            if os.path.isfile("vgg19-dcbb9e9d.pth") == False:
                # 没模型train个锤子，赶紧下载吧
                url = "http://download.pytorch.org/models/vgg19-dcbb9e9d.pth"
                cmd = "wget " + url
                if os.system(cmd):
                    # 命令行返回异常
                    import platform
                    if platform.system() == 'Linux':
                        raise Exception("Wget did not work. But alternatively you can download the model yourself.")
                    elif platform.system() == 'Windows':
                        raise Exception('You should firstly install "wget" if you run on Windows. Or you can download the model yourself.')
                    else:
                        raise Exception("Unsupported platform. You can try ubuntu 18.04 or higher.")
                else:
                    if os.path.isfile("vgg19-dcbb9e9d.pth"):
                        print("Download model successful.")
                    else:
                        # 命令行返回正常，但是文件不存在，说明网不好没下下来
                        raise Exception("Download model failed. Please check your Internet.")
            vgg_state_dict = torch.load("vgg19-dcbb9e9d.pth")
            vgg_keys = vgg_state_dict.keys()

            # 只加载VGG前10层
            weights_load = {}
            for i in range(20):
                # 因为weight + bias重复10次，所以是前20个
                weights_load[list(model.state_dict().keys())[i]
                            ] = vgg_state_dict[list(vgg_keys)[i]]

            state = model.state_dict()
            state.update(weights_load)
            model.load_state_dict(state)
            print('load Imagenet pretrained VGG model')
            return 3

def train_preparation():
    import config
    if torch.cuda.is_available():
        torch.device('cuda')
        gpu_pin_memory = True
    else:
        torch.device('cpu')
        gpu_pin_memory = False
    from lib import transforms, datasets
    preprocess = transforms.Compose([
        transforms.Normalize(),
        transforms.RandomApply(transforms.HFlip(), 0.5),
        transforms.RescaleRelative(),
        transforms.Crop(368),       # 训练时的图像大小为368
        transforms.CenterPad(368),
    ])

    train_datas = [datasets.CocoKeypoints(
        root = config.IMAGE_DIR_TRAIN,
        annFile = item,
        preprocess = preprocess,
        image_transform = transforms.image_transform_train,
        target_transforms = None,
        n_images = config.num_image_pretrain,
    ) for item in config.ANNOTATIONS_TRAIN]

    train_data = torch.utils.data.ConcatDataset(train_datas)
    
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size = config.batch_size, shuffle=True,
        pin_memory=gpu_pin_memory, num_workers = config.loader_workers, drop_last=True)

    val_data = datasets.CocoKeypoints(
        root = config.IMAGE_DIR_VAL,
        annFile = config.ANNOTATIONS_VAL,
        preprocess=preprocess,
        image_transform=transforms.image_transform_train,
        target_transforms = None,
        n_images = config.num_image_pretrain,
    )
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size = config.batch_size, shuffle=False,
        pin_memory=gpu_pin_memory, num_workers = config.loader_workers, drop_last=True)

    return train_loader, val_loader, train_data, val_data

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_loss(intermediate_map, heat_temp, vec_temp):
    import torch.nn
    import collections
    saved_for_log = collections.OrderedDict()
    criterion = torch.nn.MSELoss(reduction='mean').cuda()
    total_loss = 0

    for j in range(3):
        pred_paf = intermediate_map[j]
        loss_paf = criterion(pred_paf, vec_temp)
        total_loss += loss_paf
        saved_for_log["loss_stage%d" % (j+1)] = loss_paf.item()
    pred_hm = intermediate_map[3]
    loss_hm = criterion(pred_hm, heat_temp)
    total_loss += loss_hm
    saved_for_log["loss_stage4"] = loss_hm.item()

    saved_for_log['max_ht'] = torch.max(
        intermediate_map[-1].data[:, 0:-1, :, :]).item()
    saved_for_log['min_ht'] = torch.min(
        intermediate_map[-1].data[:, 0:-1, :, :]).item()
    saved_for_log['max_paf'] = torch.max(intermediate_map[-2].data).item()
    saved_for_log['min_paf'] = torch.min(intermediate_map[-2].data).item()

    return total_loss, saved_for_log

import time
from config import print_freq

def train(train_loader, model, optimizer, epoch):
    import torch.cuda
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    
    meter_dict = {}
    meter_dict['loss_stage1'] = AverageMeter()
    meter_dict['loss_stage2'] = AverageMeter()
    meter_dict['loss_stage3'] = AverageMeter()
    meter_dict['loss_stage4'] = AverageMeter()
    meter_dict['max_ht'] = AverageMeter()
    meter_dict['min_ht'] = AverageMeter()    
    meter_dict['max_paf'] = AverageMeter()    
    meter_dict['min_paf'] = AverageMeter()
    
    # switch to train mode
    model.train()

    end = time.time()
    for i, (img, heatmap_target, paf_target) in enumerate(train_loader):
        # measure data loading time
        #writer.add_text('Text', 'text logged at step:' + str(i), i)
        
        #for name, param in model.named_parameters():
        #    writer.add_histogram(name, param.clone().cpu().data.numpy(),i)        
        data_time.update(time.time() - end)

        # 实验室机子只有一块GTX1080，显存太少，只好不断的清理空间
        torch.cuda.empty_cache()

        img = img.cuda()
        heatmap_target = heatmap_target.cuda()
        paf_target = paf_target.cuda()
        # compute output
        intermediate_map = model(img)
        
        total_loss, saved_for_log = get_loss(intermediate_map, heatmap_target, paf_target)
        
        for name,_ in meter_dict.items():
            meter_dict[name].update(saved_for_log[name], img.size(0))
        losses.update(total_loss, img.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % print_freq == 0:
            # Print general information
            print(">> Time: " + time.strftime("%Y-%m-%d %H:%M:%S.", time.localtime()), end = "\t")
            print("[Train] Epoch %d, Iteration [%d/%d]:" % (epoch, i, len(train_loader)))
            print("   Total Loss = {loss:.6f}".format(loss=losses.val))

            """
            # Obtain the detailed losses
            # 获取具体每个stage的loss
            stage_losses_l1 = list()
            stage_losses_l2 = list()
            for j, (name, value) in enumerate(meter_dict.items()):
                if j%2:
                    stage_losses_l2.append(value.val)
                else:
                    stage_losses_l1.append(value.val)
            # L1 loss
            print_text = "   L1 loss:"
            for j in range(len(stage_losses_l1)):
                print_text += "%8.4f" % stage_losses_l1[j]
            print(print_text)
            # L2 loss
            print_text = "   L2 loss:"
            for j in range(len(stage_losses_l2)):
                print_text += "%8.4f" % stage_losses_l2[j]
            print(print_text)
            """
    return losses.avg

def validate(val_loader, model, epoch):
    import torch.cuda
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    
    meter_dict = {}
    meter_dict['loss_stage1'] = AverageMeter()
    meter_dict['loss_stage2'] = AverageMeter()
    meter_dict['loss_stage3'] = AverageMeter()
    meter_dict['loss_stage4'] = AverageMeter()
    meter_dict['max_ht'] = AverageMeter()
    meter_dict['min_ht'] = AverageMeter()    
    meter_dict['max_paf'] = AverageMeter()    
    meter_dict['min_paf'] = AverageMeter()
    # switch to train mode
    model.eval()

    end = time.time()
    for i, (img, heatmap_target, paf_target) in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # Recycle the GPU memory since my GTX1080 card is limited
        # 实验室机子只有一块GTX1080，显存太少，只好不断的清理空间
        torch.cuda.empty_cache()

        img = img.cuda()
        heatmap_target = heatmap_target.cuda()
        paf_target = paf_target.cuda()
        
        # compute output
        intermediate_map = model(img)
        
        total_loss, saved_for_log = get_loss(intermediate_map, heatmap_target, paf_target)
               
        #for name,_ in meter_dict.items():
        #    meter_dict[name].update(saved_for_log[name], img.size(0))
            
        losses.update(total_loss.item(), img.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()  
        if i % print_freq == 0:
            # Print general information
            # Validation阶段没有各个stage loss的计算
            print(">  Time: " + time.strftime("%Y-%m-%d %H:%M:%S.", time.localtime()), end = "\t")
            print("[Validation] Epoch %d, Iteration [%d/%d]" % (epoch, i, len(val_loader)), end = ":")
            print("Loss = {loss:.6f}".format(loss=losses.val))

    return losses.avg