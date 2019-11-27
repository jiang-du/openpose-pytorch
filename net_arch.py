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
                layers += [conv2d, torch.nn.ReLU(inplace=True)]
    one_ = list(cfg_dict[-1].keys())
    k = one_[0]
    v = cfg_dict[-1][k]
    conv2d = torch.nn.Conv2d(in_channels=v[0], out_channels=v[1],
                       kernel_size=v[2], stride=v[3], padding=v[4])
    layers += [conv2d]
    return torch.nn.Sequential(*layers)

def openpose_model(num_stages = 6):
    if num_stages>6:
        num_stages = 6
        raise Exception("At most 6 stages.")
    blocks = {}
    # block0是用了VGG前10层作为特征提取，这部分需要载入ImageNet权重
    block0 = [{'conv1_1': [3, 64, 3, 1, 1]},
                {'conv1_2': [64, 64, 3, 1, 1]},
                {'pool1_stage1': [2, 2, 0]},
                {'conv2_1': [64, 128, 3, 1, 1]},
                {'conv2_2': [128, 128, 3, 1, 1]},
                {'pool2_stage1': [2, 2, 0]},
                {'conv3_1': [128, 256, 3, 1, 1]},
                {'conv3_2': [256, 256, 3, 1, 1]},
                {'conv3_3': [256, 256, 3, 1, 1]},
                {'conv3_4': [256, 256, 3, 1, 1]},
                {'pool3_stage1': [2, 2, 0]},
                {'conv4_1': [256, 512, 3, 1, 1]},
                {'conv4_2': [512, 512, 3, 1, 1]},
                {'conv4_3_CPM': [512, 256, 3, 1, 1]},
                {'conv4_4_CPM': [256, 128, 3, 1, 1]}]

    # Stage 1
    blocks['block1_1'] = [{'conv5_1_CPM_L1': [128, 128, 3, 1, 1]},
                          {'conv5_2_CPM_L1': [128, 128, 3, 1, 1]},
                          {'conv5_3_CPM_L1': [128, 128, 3, 1, 1]},
                          {'conv5_4_CPM_L1': [128, 512, 1, 1, 0]},
                          {'conv5_5_CPM_L1': [512, 38, 1, 1, 0]}]

    blocks['block1_2'] = [{'conv5_1_CPM_L2': [128, 128, 3, 1, 1]},
                          {'conv5_2_CPM_L2': [128, 128, 3, 1, 1]},
                          {'conv5_3_CPM_L2': [128, 128, 3, 1, 1]},
                          {'conv5_4_CPM_L2': [128, 512, 1, 1, 0]},
                          {'conv5_5_CPM_L2': [512, 19, 1, 1, 0]}]

    # Stages 2 - num_stages (默认6)
    for i in range(2, num_stages + 1):
        blocks['block%d_1' % i] = [
            {'Mconv1_stage%d_L1' % i: [185, 128, 7, 1, 3]},
            {'Mconv2_stage%d_L1' % i: [128, 128, 7, 1, 3]},
            {'Mconv3_stage%d_L1' % i: [128, 128, 7, 1, 3]},
            {'Mconv4_stage%d_L1' % i: [128, 128, 7, 1, 3]},
            {'Mconv5_stage%d_L1' % i: [128, 128, 7, 1, 3]},
            {'Mconv6_stage%d_L1' % i: [128, 128, 1, 1, 0]},
            {'Mconv7_stage%d_L1' % i: [128, 38, 1, 1, 0]}
        ]

        blocks['block%d_2' % i] = [
            {'Mconv1_stage%d_L2' % i: [185, 128, 7, 1, 3]},
            {'Mconv2_stage%d_L2' % i: [128, 128, 7, 1, 3]},
            {'Mconv3_stage%d_L2' % i: [128, 128, 7, 1, 3]},
            {'Mconv4_stage%d_L2' % i: [128, 128, 7, 1, 3]},
            {'Mconv5_stage%d_L2' % i: [128, 128, 7, 1, 3]},
            {'Mconv6_stage%d_L2' % i: [128, 128, 1, 1, 0]},
            {'Mconv7_stage%d_L2' % i: [128, 19, 1, 1, 0]}
        ]

    models = {}

    print("Bulding VGG19...")
    models['block0'] = make_vgg19_block(block0)

    for k, v in blocks.items():
        models[k] = make_stages(list(v))

    class cmu_openpose_model(torch.nn.Module):
        def __init__(self, model_dict):
            super(cmu_openpose_model, self).__init__()
            self.model0 = model_dict['block0']
            self.model1_1 = model_dict['block1_1']
            # 采用动态执行代码的方式，构建任意stage的网络模型
            for i in range(2, num_stages + 1):
                exec("self.model%d_1 = model_dict['block%d_1']"%(i, i))
            """
            if num_stages > 1:
                self.model2_1 = model_dict['block2_1']
            if num_stages > 2:
                self.model3_1 = model_dict['block3_1']
            if num_stages > 3:
                self.model4_1 = model_dict['block4_1']
            if num_stages > 4:
                self.model5_1 = model_dict['block5_1']
            if num_stages > 5:
                self.model6_1 = model_dict['block6_1']
            """
            
            self.model1_2 = model_dict['block1_2']
            # 采用动态执行代码的方式，构建任意stage的网络模型(同上)
            for i in range(2, num_stages + 1):
                exec("self.model%d_2 = model_dict['block%d_2']"%(i, i))
            """
            if num_stages > 1:
                self.model2_2 = model_dict['block2_2']
            if num_stages > 2:
                self.model3_2 = model_dict['block3_2']
            if num_stages > 3:
                self.model4_2 = model_dict['block4_2']
            if num_stages > 4:
                self.model5_2 = model_dict['block5_2']
            if num_stages > 5:
                self.model6_2 = model_dict['block6_2']
            """

            self._initialize_weights_norm()

        def forward(self, x):

            saved_for_loss = []
            out0 = self.model0(x)

            out1_1 = self.model1_1(out0)
            out1_2 = self.model1_2(out0)
            out2 = torch.cat([out1_1, out1_2, out0], 1)
            saved_for_loss.append(out1_1)
            saved_for_loss.append(out1_2)

            """
            if num_stages > 1:
                out2_1 = self.model2_1(out2)
                out2_2 = self.model2_2(out2)
                out3 = torch.cat([out2_1, out2_2, out0], 1)
                saved_for_loss.append(out2_1)
                saved_for_loss.append(out2_2)
            else:
                return (out1_1, out1_2), saved_for_loss

            if num_stages > 2:
                out3_1 = self.model3_1(out3)
                out3_2 = self.model3_2(out3)
                out4 = torch.cat([out3_1, out3_2, out0], 1)
                saved_for_loss.append(out3_1)
                saved_for_loss.append(out3_2)
            else:
                return (out2_1, out2_2), saved_for_loss

            if num_stages > 3:
                out4_1 = self.model4_1(out4)
                out4_2 = self.model4_2(out4)
                out5 = torch.cat([out4_1, out4_2, out0], 1)
                saved_for_loss.append(out4_1)
                saved_for_loss.append(out4_2)
            else:
                return (out3_1, out3_2), saved_for_loss

            if num_stages > 4:
                out5_1 = self.model5_1(out5)
                out5_2 = self.model5_2(out5)
                out6 = torch.cat([out5_1, out5_2, out0], 1)
                saved_for_loss.append(out5_1)
                saved_for_loss.append(out5_2)
            else:
                return (out4_1, out4_2), saved_for_loss

            if num_stages > 5:
                out6_1 = self.model6_1(out6)
                out6_2 = self.model6_2(out6)
                saved_for_loss.append(out6_1)
                saved_for_loss.append(out6_2)
            else:
                return (out5_1, out5_2), saved_for_loss

            return (out6_1, out6_2), saved_for_loss
            """

            # i的取值范围是2到stage个数
            for i in range(2, num_stages + 1):
                exec("out%d_1 = self.model%d_1(out%d)"%(i, i, i))
                exec("out%d_2 = self.model%d_2(out%d)"%(i, i, i))
                if i < num_stages:
                    exec("out%d = torch.cat([out%d_1, out%d_2, out0], 1)"%(i+1, i, i))
                exec("saved_for_loss.append(out%d_1)"%i)
                exec("saved_for_loss.append(out%d_2)"%i)
            # 不能在exec函数中调用return，否则报错"SyntaxError: 'return' outside function"
            #   exec("return (out%d_1, out%d_2), saved_for_loss"%(num_stages, num_stages))
            # 因此改用eval动态赋值的方式
            return (eval("out%d_1"%num_stages), eval("out%d_2"%num_stages)), saved_for_loss

        def _initialize_weights_norm(self):

            for m in self.modules():
                if isinstance(m, torch.nn.Conv2d):
                    torch.nn.init.normal_(m.weight, std=0.01)
                    # 预留给mobilenet模型用(如果后面想起来写的话)，因为他的conv2d没有bias
                    if m.bias is not None:
                        torch.nn.init.constant_(m.bias, 0.0)

            # 每个stage最后一层没有Relu
            torch.nn.init.normal_(self.model1_1[8].weight, std=0.01)
            torch.nn.init.normal_(self.model1_2[8].weight, std=0.01)

            for i in range(2, num_stages + 1):
                exec("torch.nn.init.normal_(self.model%d_1[12].weight, std=0.01)"%i)
                exec("torch.nn.init.normal_(self.model%d_2[12].weight, std=0.01)"%i)

            """
            if num_stages > 1:
                torch.nn.init.normal_(self.model2_1[12].weight, std=0.01)
                torch.nn.init.normal_(self.model2_2[12].weight, std=0.01)
            if num_stages > 2:
                torch.nn.init.normal_(self.model3_1[12].weight, std=0.01)
                torch.nn.init.normal_(self.model3_2[12].weight, std=0.01)
            if num_stages > 3:
                torch.nn.init.normal_(self.model4_1[12].weight, std=0.01)
                torch.nn.init.normal_(self.model4_2[12].weight, std=0.01)
            if num_stages > 4:
                torch.nn.init.normal_(self.model5_1[12].weight, std=0.01)
                torch.nn.init.normal_(self.model5_2[12].weight, std=0.01)
            if num_stages > 5:
                torch.nn.init.normal_(self.model6_1[12].weight, std=0.01)
                torch.nn.init.normal_(self.model6_2[12].weight, std=0.01)
            """

    model = cmu_openpose_model(models)
    return model
