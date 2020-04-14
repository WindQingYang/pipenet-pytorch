'''

Pytorch code for 'Chalearn Multi-modal Cross-ethnicity Face anti-spoofing Recognition Challenge@CVPR2020'
By Qing Yang, 2020/03/01

MIT License
Copyright (c) 2019

'''
import os
from utils import *
from torchvision.models.resnet import BasicBlock
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

BatchNorm2d = nn.BatchNorm2d
import torchvision.models as tvm
from models.backbone.SENet import SEModule, SENet ,SEBottleneck, SEResNetBottleneck, SEResNeXtBottleneck
from models.backbone.Xception import Xception



###########################################################################################3
class PipeNet(nn.Module):
    def load_pretrain(self, pretrain_file):
        #raise NotImplementedError
        pretrain_state_dict = torch.load(pretrain_file)
        state_dict = self.state_dict()
        keys = list(state_dict.keys())
        for key in keys:
            state_dict[key] = pretrain_state_dict[key]

        self.load_state_dict(state_dict)
        print('')


    def __init__(self, num_class=2):
        super(PipeNet,self).__init__()

        ###Pipeline elements:
        self.is_first_bn = True
        self.first_bn = nn.BatchNorm2d(3)

        #resnet18 encoder#
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)
        self.resnet18_encoder  = tvm.resnet18(pretrained=True)
        self.resnet18_conv1 = nn.Sequential(self.resnet18_encoder.conv1,
                                   self.resnet18_encoder.bn1,
                                   self.resnet18_encoder.relu,
                                   self.pool)

        self.resnet18_conv2 = self.resnet18_encoder.layer1
        self.resnet18_conv3 = self.resnet18_encoder.layer2
        self.resnet18_conv4 = self.resnet18_encoder.layer3

        #Xception encoder#
        self.xception_encoder = Xception(num_class=1000)

        self.xception_conv1 = self.xception_encoder.conv1
        self.xception_conv2 = self.xception_encoder.conv2
        self.xception_blocks = self.xception_encoder.blocks2

        #ResNext encoder
        self.senet154_encoder  = SENet(SEBottleneck, [1, 1, 1, 1], groups=64, reduction=16,
                      dropout_p=0.2, inplanes=128, input_3x3=True,
                      downsample_kernel_size=3, downsample_padding=1,
                      num_class=1000)
        self.seresnext_encoder  = SENet(SEResNeXtBottleneck, [2, 2, 2, 2], groups=32, reduction=16,
                      dropout_p=None, inplanes=64, input_3x3=False,
                      downsample_kernel_size=1, downsample_padding=0,
                      num_class=1000)
        self.seresnext2_encoder  = SENet(SEResNeXtBottleneck, [3, 4, 4, 3], groups=32, reduction=16,
                      dropout_p=None, inplanes=64, input_3x3=False,
                      downsample_kernel_size=1, downsample_padding=0,
                      num_class=1000)
        # self.seresnet_encoder  = SENet(SEResNetBottleneck, [1, 4, 4, 1], groups=1, reduction=16,
        #               dropout_p=None, inplanes=64, input_3x3=False,
        #               downsample_kernel_size=1, downsample_padding=0,
        #               num_class=1000)

        self.seresnext_conv1 = self.seresnext_encoder.layer0
        self.seresnext_conv2 = self.seresnext_encoder.layer1
        self.seresnext_conv3 = self.seresnext_encoder.layer2

        ###

        self.color_SE = SEModule(128,reduction=16)
        self.depth_SE = SEModule(128,reduction=16)
        self.ir_SE = SEModule(128,reduction=16)

        self.res_0 = self._make_layer(BasicBlock, 128*3, 256, 2, stride=2)
        self.res_1 = self._make_layer(BasicBlock, 256, 512, 2, stride=2)

        self.fc = nn.Sequential(nn.Dropout(0.5),
                                nn.Linear(512, 256),
                                nn.ReLU(inplace=True),
                                nn.Linear(256, num_class))


    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 :
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),)

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def color_forward(self, x):
        batch_size, C, H, W = x.shape

        if self.is_first_bn:
            x = self.first_bn(x)
        else:
            mean = [0.485, 0.456, 0.406]  # rgb
            std = [0.229, 0.224, 0.225]

            x = torch.cat([
                (x[:, [0]] - mean[0]) / std[0],
                (x[:, [1]] - mean[1]) / std[1],
                (x[:, [2]] - mean[2]) / std[2],
            ], 1)

        x = self.resnet18_conv1(x)#;print('c1', x.size())
        x = self.resnet18_conv2(x)#;print('c2', x.size())
        x = self.resnet18_conv3(x)#;print('c3', x.size())
        # x = self.resnext_conv1(x)#;print('i1', x.size())
        # x = self.resnext_conv2(x)#;print('i2', x.size())
        # x = self.resnext_conv3(x)#;print('i3', x.size())

        return x

    def depth_forward(self, x):
        batch_size, C, H, W = x.shape

        if self.is_first_bn:
            x = self.first_bn(x)
        else:
            mean = [0.485, 0.456, 0.406]  # rgb
            std = [0.229, 0.224, 0.225]

            x = torch.cat([
                (x[:, [0]] - mean[0]) / std[0],
                (x[:, [1]] - mean[1]) / std[1],
                (x[:, [2]] - mean[2]) / std[2],
            ], 1)

        x = self.resnet18_conv1(x)#;print('c1', x.size())
        x = self.resnet18_conv2(x)#;print('c2', x.size())
        x = self.resnet18_conv3(x)#;print('c3', x.size())
        # x = self.xception_conv1(x)#;print('d1', x.size())
        # x = self.xception_conv2(x)#;print('d2', x.size())
        # x = self.xception_blocks(x)#;print('d3', x.size())

        return x

    def ir_forward(self, x):
        batch_size, C, H, W = x.shape

        if self.is_first_bn:
            x = self.first_bn(x)
        else:
            mean = [0.485, 0.456, 0.406]  # rgb
            std = [0.229, 0.224, 0.225]

            x = torch.cat([
                (x[:, [0]] - mean[0]) / std[0],
                (x[:, [1]] - mean[1]) / std[1],
                (x[:, [2]] - mean[2]) / std[2],
            ], 1)

        x = self.resnet18_conv1(x)#;print('c1', x.size())
        x = self.resnet18_conv2(x)#;print('c2', x.size())
        x = self.resnet18_conv3(x)#;print('c3', x.size())
        # x = self.seresnext_conv1(x)#;print('i1', x.size())
        # x = self.seresnext_conv2(x)#;print('i2', x.size())
        # x = self.seresnext_conv3(x)#;print('i3', x.size())

        return x


    def forward(self, x):
        batch_size,C,H,W = x.shape

        color = x[:, 0:3,:,:]
        depth = x[:, 3:6,:,:]
        ir = x[:, 6:9,:,:]

        color_feas = self.color_forward(color) #; print('cfea',color_feas.size())
        depth_feas = self.depth_forward(depth) #; print('dfea',depth_feas.size())
        ir_feas = self.ir_forward(ir) #; print('ifea',ir_feas.size())

        color_feas = self.color_SE(color_feas) #; print('csefea',color_feas.size())
        depth_feas = self.depth_SE(depth_feas) #; print('dsefea',depth_feas.size())
        ir_feas = self.ir_SE(ir_feas) #; print('isefea',ir_feas.size())

        fea = torch.cat([color_feas, depth_feas, ir_feas], dim=1) #; print('cat',fea.size())

        x = self.res_0(fea) #; print('res0',x.size())
        x = self.res_1(x) #; print('res1',x.size())
        x = F.adaptive_avg_pool2d(x, output_size=1).view(batch_size, -1)
        x = self.fc(x)
        return x

    def set_mode(self, mode, is_freeze_bn=False ):
        self.mode = mode
        if mode in ['eval', 'valid', 'test']:
            self.eval()
        elif mode in ['backup']:
            self.train()
            if is_freeze_bn==True: ##freeze
                for m in self.modules():
                    if isinstance(m, BatchNorm2d):
                        m.eval()
                        m.weight.requires_grad = False
                        m.bias.requires_grad   = False

### run ##############################################################################
def run_check_net():
    num_class = 2
    net = Net(num_class)
    print(net)

########################################################################################
if __name__ == '__main__':
    import os
    run_check_net()
    print( 'sucessful!')