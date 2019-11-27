# 加载COCO数据集
import functions
train_loader, val_loader, train_data, val_data = functions.train_preparation()

# 定义网络结构，并加载初始权重
import torch
from net_arch import openpose_model
import config
model = openpose_model(num_stages = config.num_stages).cuda()
# print(model)

# 机子只有一块卡，不需要并行计算
# model = torch.nn.DataParallel(model).cuda()

# 加载预训练权重
functions.load_weights(model)

# 前5个固定epoch固定住VGG部分，并训练后面的网络结构
for i in range(20):
    for param in model.model0[i].parameters():
        # 梯度锁定
        param.requires_grad = False

trainable_vars = [param for param in model.parameters() if param.requires_grad]
optimizer = torch.optim.SGD(trainable_vars, lr = config.learning_rate,
                           momentum = config.momentum,
                           weight_decay = config.weight_decay,
                           nesterov = config.nesterov)

for epoch in range(5):
    # train
    train_loss = functions.train(train_loader, model, optimizer, epoch)
    torch.cuda.empty_cache()

    # validation
    val_loss = functions.validate(val_loader, model, epoch)
    torch.cuda.empty_cache()

    # 训练前5个epoch时保存临时权重
    torch.save(model.state_dict(), config.pre_model_name)
    print("Saved the weight of epoch %d."%epoch)

# 对VGG部分的梯度解除锁定
for param in model.parameters():
    param.requires_grad = True

trainable_vars = [param for param in model.parameters() if param.requires_grad]
optimizer = torch.optim.SGD(trainable_vars, lr = config.learning_rate,
                           momentum = config.momentum,
                           weight_decay = config.weight_decay,
                           nesterov = config.nesterov)

# 定义learning rate调整规则
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=5, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=3, min_lr=0, eps=1e-08)

from numpy import inf
best_val_loss = inf

for epoch in range(5, config.num_epochs):
    # train
    train_loss = functions.train(train_loader, model, optimizer, epoch)

    # validation
    val_loss = functions.validate(val_loader, model, epoch)
    
    lr_scheduler.step(val_loss)
    
    if (val_loss < best_val_loss):
        # 更新当前最佳
        best_val_loss = val_loss
        # 如果val loss更小了，保存模型
        torch.save(model.state_dict(), config.model_save_filename)