import torch

def make_vgg19_block(block):
    """Builds a vgg19 block from a dictionary
    Args:
        block: a dictionary
    """
    layers = []
    for i in range(len(block)):
        one_ = block[i]
        for k, v in one_.items():
            if 'pool' in k:
                layers += [torch.nn.MaxPool2d(kernel_size=v[0], stride=v[1],
                                        padding=v[2])]
            else:
                conv2d = torch.nn.Conv2d(in_channels=v[0], out_channels=v[1],
                                   kernel_size=v[2], stride=v[3],
                                   padding=v[4])
                layers += [conv2d, torch.nn.ReLU(inplace=True)]
    return torch.nn.Sequential(*layers)

def make_stages(cfg_dict):
    """Builds CPM stages from a dictionary
    Args:
        cfg_dict: a dictionary
    """
    stage_layer = torch.nn.ModuleList()
    layers = []
    for i in range(len(cfg_dict) - 1):
        one_ = cfg_dict[i]
        for k, v in one_.items():
            if 'pool' in k:
                layers += [torch.nn.MaxPool2d(kernel_size=v[0], stride=v[1],
                                        padding=v[2])]
            else:
                conv2d = torch.nn.Conv2d(in_channels=v[0], out_channels=v[1],
                                   kernel_size=v[2], stride=v[3],
                                   padding=v[4])
                stage_layer.append(conv2d)
                stage_layer.append(torch.nn.ReLU(inplace=True))
    one_ = list(cfg_dict[-1].keys())
    k = one_[0]
    v = cfg_dict[-1][k]
    conv2d = torch.nn.Conv2d(in_channels=v[0], out_channels=v[1],
                       kernel_size=v[2], stride=v[3], padding=v[4])
    stage_layer.append(conv2d)
    return stage_layer # 注意这里不能用Sequential，因为要写densenet

def openpose_pami(num_stages = 6):
    # tp表示PAF场的stage，tc表示confidence map的stage数
    tp=3
    tc=1 # 暂时还不能编辑
    blocks = {}
    # block0是用了VGG前10层作为特征提取，这部分需要载入ImageNet权重
    block0 = [{'conv1_1': [3, 64, 3, 1, 1]},
                {'conv1_2': [64, 64, 3, 1, 1]},
                {'pool1': [2, 2, 0]},
                {'conv2_1': [64, 128, 3, 1, 1]},
                {'conv2_2': [128, 128, 3, 1, 1]},
                {'pool2': [2, 2, 0]},
                {'conv3_1': [128, 256, 3, 1, 1]},
                {'conv3_2': [256, 256, 3, 1, 1]},
                {'conv3_3': [256, 256, 3, 1, 1]},
                {'conv3_4': [256, 256, 3, 1, 1]},
                {'pool3': [2, 2, 0]},
                {'conv4_1': [256, 512, 3, 1, 1]},
                {'conv4_2': [512, 512, 3, 1, 1]},
                {'conv4_3': [512, 256, 3, 1, 1]},
                {'conv4_4': [256, 128, 3, 1, 1]}]

    # Stage 1 提PAF场所以输出heatmap是输出channel=38的维度
    blocks['block1'] = [
        {'conv_stage1_denseblock1_1': [128, 128, 3, 1, 1]},
        {'conv_stage1_denseblock1_2': [128, 128, 3, 1, 1]},
        {'conv_stage1_denseblock1_3': [128, 128, 3, 1, 1]},
        {'conv_stage1_denseblock2_1': [128, 128, 3, 1, 1]},
        {'conv_stage1_denseblock2_2': [128, 128, 3, 1, 1]},
        {'conv_stage1_denseblock2_3': [128, 128, 3, 1, 1]},
        {'conv_stage1_denseblock3_1': [128, 128, 3, 1, 1]},
        {'conv_stage1_denseblock3_2': [128, 128, 3, 1, 1]},
        {'conv_stage1_denseblock3_3': [128, 128, 3, 1, 1]},
        {'conv_stage1_denseblock4_1': [128, 128, 3, 1, 1]},
        {'conv_stage1_denseblock4_2': [128, 128, 3, 1, 1]},
        {'conv_stage1_denseblock4_3': [128, 128, 3, 1, 1]},
        {'conv_stage1_denseblock5_1': [128, 128, 3, 1, 1]},
        {'conv_stage1_denseblock5_2': [128, 128, 3, 1, 1]},
        {'conv_stage1_denseblock5_3': [128, 128, 3, 1, 1]},
        {'conv_stage1_6': [128, 512, 1, 1, 0]},
        {'conv_stage1_7': [512, 38, 1, 1, 0]}
    ]

    # Stage 2 - tp 提PAF场所以输出heatmap是输出channel=38的维度
    for i in range(2, tp+1):
        blocks['block%d' % i] = [
            {'conv_stage%d_denseblock1_1' % i: [128+38, 128, 3, 1, 1]},
            {'conv_stage%d_denseblock1_2' % i: [128, 128, 3, 1, 1]},
            {'conv_stage%d_denseblock1_3' % i: [128, 128, 3, 1, 1]},
            {'conv_stage%d_denseblock2_1' % i: [128, 128, 3, 1, 1]},
            {'conv_stage%d_denseblock2_2' % i: [128, 128, 3, 1, 1]},
            {'conv_stage%d_denseblock2_3' % i: [128, 128, 3, 1, 1]},
            {'conv_stage%d_denseblock3_1' % i: [128, 128, 3, 1, 1]},
            {'conv_stage%d_denseblock3_2' % i: [128, 128, 3, 1, 1]},
            {'conv_stage%d_denseblock3_3' % i: [128, 128, 3, 1, 1]},
            {'conv_stage%d_denseblock4_1' % i: [128, 128, 3, 1, 1]},
            {'conv_stage%d_denseblock4_2' % i: [128, 128, 3, 1, 1]},
            {'conv_stage%d_denseblock4_3' % i: [128, 128, 3, 1, 1]},
            {'conv_stage%d_denseblock5_1' % i: [128, 128, 3, 1, 1]},
            {'conv_stage%d_denseblock5_2' % i: [128, 128, 3, 1, 1]},
            {'conv_stage%d_denseblock5_3' % i: [128, 128, 3, 1, 1]},
            {'conv_stage%d_6' % i: [128, 128, 1, 1, 0]},
            {'conv_stage%d_7' % i: [128, 38, 1, 1, 0]}
        ]

    # Stage tp+1 (tc=1) 提confidence map所以输出heatmap是输出channel=19的维度
    for i in range(tp+1, tp+tc+1):
        blocks['block%d' % i] = [
            {'conv_stage%d_denseblock1_1' % i: [128+38, 128, 3, 1, 1]},
            {'conv_stage%d_denseblock1_2' % i: [128, 128, 3, 1, 1]},
            {'conv_stage%d_denseblock1_3' % i: [128, 128, 3, 1, 1]},
            {'conv_stage%d_denseblock2_1' % i: [128, 128, 3, 1, 1]},
            {'conv_stage%d_denseblock2_2' % i: [128, 128, 3, 1, 1]},
            {'conv_stage%d_denseblock2_3' % i: [128, 128, 3, 1, 1]},
            {'conv_stage%d_denseblock3_1' % i: [128, 128, 3, 1, 1]},
            {'conv_stage%d_denseblock3_2' % i: [128, 128, 3, 1, 1]},
            {'conv_stage%d_denseblock3_3' % i: [128, 128, 3, 1, 1]},
            {'conv_stage%d_denseblock4_1' % i: [128, 128, 3, 1, 1]},
            {'conv_stage%d_denseblock4_2' % i: [128, 128, 3, 1, 1]},
            {'conv_stage%d_denseblock4_3' % i: [128, 128, 3, 1, 1]},
            {'conv_stage%d_denseblock5_1' % i: [128, 128, 3, 1, 1]},
            {'conv_stage%d_denseblock5_2' % i: [128, 128, 3, 1, 1]},
            {'conv_stage%d_denseblock5_3' % i: [128, 128, 3, 1, 1]},
            {'conv_stage%d_6' % i: [128, 128, 1, 1, 0]},
            {'conv_stage%d_7' % i: [128, 19, 1, 1, 0]}
        ]

    models = {}

    models['block0'] = make_vgg19_block(block0)

    for k, v in blocks.items():
        models[k] = make_stages(list(v))

    # 构造网络模型
    class cmu_openpose_model(torch.nn.Module):
        def __init__(self, model_dict):
            super(cmu_openpose_model, self).__init__()
            self.model0 = model_dict['block0']
            # 采用动态执行代码的方式，构建任意stage位tp+tc的网络模型
            for i in range(1, tp+tc+1):
                exec("self.model%d = model_dict['block%d']"%(i, i))
            # 随机初始化
            self._initialize_weights_norm()

        def forward(self, x):
            # 定义intermediate_map用来存放每个stage的map信息
            intermediate_map = []
            # out0用来存储VGG19的前10层信息
            out0 = self.model0(x)

            # 实现stages
            for i_stage in range(1, tp+tc+1):
                if i_stage > 1:
                    # 除了第一个stage以外，每个stage使用上一个stage的结果加上VGG feature
                    out1 = torch.cat([out0, out1], 1)
                else:
                    # 第一个stage只用VGG feature
                    out1 = out0

                # 对连续5个块的densenet的实现
                for i in range(5):
                    # out1_d1 = self.model1[i*3+0](out1)
                    out1_d1 = eval("self.model%d[i*6+0](out1)" % i_stage)
                    out1_d1 = eval("self.model%d[i*6+1](out1_d1)" % i_stage)
                    # out1_d2 = self.model1[i*3+1](out1_d1)
                    out1_d2 = eval("self.model%d[i*6+2](out1_d1)" % i_stage)
                    out1_d2 = eval("self.model%d[i*6+3](out1_d2)" % i_stage)
                    # out1_d3 = self.model1[i*3+2](out1_d2)
                    out1_d3 = eval("self.model%d[i*6+4](out1_d2)" % i_stage)
                    out1_d3 = eval("self.model%d[i*6+5](out1_d3)" % i_stage)
                    out1 = out1_d1 + out1_d2 + out1_d3
                
                # 然后是2个1x1整合
                # out1 = self.model1[31](self.model1[30](out1))
                out1 = eval("self.model%d[31](self.model%d[30](out1))" % (i_stage, i_stage))
                # out1 = self.model1[32](out1) # no relu
                out1 = eval("self.model%d[32](out1)" % i_stage)
                # 这里out1得到的就是PAF或者confidence map

                # 记录下这个map用于intermediate supervision
                intermediate_map.append(out1)
            # 每个stage的实现到此结束


            # 不能在exec函数中调用return，否则报错"SyntaxError: 'return' outside function"
            #   exec("return (out%d_1, out%d_2), saved_for_loss"%(num_stages, num_stages))
            # 因此改用eval动态赋值的方式
            #(eval("out%d_1"%num_stages), eval("out%d_2"%num_stages)), saved_for_loss
            return intermediate_map

        def _initialize_weights_norm(self):

            for m in self.modules():
                if isinstance(m, torch.nn.Conv2d):
                    torch.nn.init.normal_(m.weight, std=0.01)
                    # 预留给mobilenet模型用(如果后面想起来写的话)，因为他的conv2d没有bias
                    if m.bias is not None:
                        torch.nn.init.constant_(m.bias, 0.0)

            # 每个stage最后一层没有Relu
            torch.nn.init.normal_(self.model1[32].weight, std=0.01)

            for i in range(2, 4):
                exec("torch.nn.init.normal_(self.model%d[32].weight, std=0.01)"%i)

    model = cmu_openpose_model(models)
    print("Openpose model building finished.")
    return model
