#!/usr/bin/python3
#coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
import torch.utils.model_zoo as model_zoo
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Bottle2neck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=26, scale=4, stype='normal'):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1
            baseWidth: basic width of conv3x3
            scale: number of scale.
            type: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(Bottle2neck, self).__init__()

        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width * scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype == 'normal':
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype == 'stage':
            out = torch.cat((out, self.pool(spx[self.nums])), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Res2Net(nn.Module):

    def __init__(self, block, layers, baseWidth=26, scale=4, num_classes=1000):
        self.inplanes = 64
        super(Res2Net, self).__init__()
        self.baseWidth = baseWidth
        self.scale = scale
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False)
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=stride, stride=stride,
                             ceil_mode=True, count_include_pad=False),
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                            stype='stage', baseWidth=self.baseWidth, scale=self.scale))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, baseWidth=self.baseWidth, scale=self.scale))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)    # torch.Size([1, 64, 176, 176])
        x = self.maxpool(x)

        x = self.layer1(x)  # torch.Size([1, 256, 88, 88])
        x = self.layer2(x)  # torch.Size([1, 512, 44, 44])
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def res2net50_v1b_26w_4s(pretrained=False, **kwargs):
    """Constructs a Res2Net-50_v1b_26w_4s lib.
    Args:
        pretrained (bool): If True, returns a lib pre-trained on ImageNet
    """
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth=26, scale=4, **kwargs)
    if pretrained:
        model_state = torch.load("/home/wuzhicong/COD/CODv162/pre-train/res2net50_v1b_26w_4s-3cf99910.pth")
        # model.load_state_dict(model_state, strict=False)
        model.load_state_dict(model_state)
    return model

# Detail block
class Xd_block(nn.Module):
    def __init__(self):
        super(Xd_block, self).__init__()

        # for encoder1
        self.conv1_1 = nn.Conv2d(64, 64, kernel_size=1)

        # for encoder3
        self.conv1_3 = nn.Conv2d(64, 64, kernel_size=1)

        # after add (encoder1\2\3)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # after add (encoder2)
        # self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        # self.bn3 = nn.BatchNorm2d(64)

        # for Y
        self.conv4 = nn.Conv2d(64, 64, kernel_size=1)

        # after concat (y)
        self.conv5_1_1 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv5_1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)  # for y5
        self.bn5_1_1 = nn.BatchNorm2d(64)
        self.bn5_1_2 = nn.BatchNorm2d(64)

        self.conv5_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn5_2 = nn.BatchNorm2d(64)
        self.conv5_3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn5_3 = nn.BatchNorm2d(64)


    def forward(self, encoder2, encoder1=None, encoder3=None, Y=None):
        # encoder1: encoder n-1 layer
        # encoder2: encoder n   layer
        # encoder3: encoder n+1 layer
        # Y: Y n-1 block

        encoder2_size = encoder2.size()[2:]

        if encoder1 is None:
            encoder1_conv = torch.zeros_like(encoder2)
        else:
            encoder1_down = F.interpolate(encoder1, size=encoder2_size, mode='bilinear')
            encoder1_conv = self.conv1_1(encoder1_down)

        if encoder3 is None:
            encoder3_conv = torch.zeros_like(encoder2)
        else:
            encoder3_up = F.interpolate(encoder3, size=encoder2_size, mode='bilinear')
            encoder3_conv = self.conv1_3(encoder3_up)

        AfterAdd1_conv = F.relu(self.bn2(self.conv2(encoder2 + encoder1_conv + encoder3_conv)), inplace=True)

        AfterAdd2 = AfterAdd1_conv + encoder2

        if Y is None:
            AfterCat_conv1 = F.relu(self.bn5_1_2(self.conv5_1_2(AfterAdd2)), inplace=True)
        else:
            Y_conv = self.conv4(F.interpolate(Y, size=encoder2_size, mode='bilinear'))
            AfterCat_conv1 = F.relu(self.bn5_1_1(self.conv5_1_1(torch.cat([AfterAdd2, Y_conv], dim=1))), inplace=True)

        AfterCat_conv2 = F.relu(self.bn5_2(self.conv5_2(AfterCat_conv1)), inplace=True)
        AfterCat_conv3 = F.relu(self.bn5_3(self.conv5_3(AfterCat_conv2)), inplace=True)

        out = AfterCat_conv3
        return out

# Body block
class Xb_block(nn.Module):
    def __init__(self):
        super(Xb_block, self).__init__()

        # for encoder1
        self.conv1_1 = nn.Conv2d(64, 64, kernel_size=1)

        # for encoder3
        self.conv1_3 = nn.Conv2d(64, 64, kernel_size=1)

        # after add (encoder1\2\3)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # after add (encoder2)
        # self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        # self.bn3 = nn.BatchNorm2d(64)

        # for Y
        self.conv4 = nn.Conv2d(64, 64, kernel_size=1)

        # after concat (y)
        self.conv5_1_1 = nn.Conv2d(192, 64, kernel_size=3, stride=1, padding=1)
        self.conv5_1_2 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)  # for y5
        self.bn5_1_1 = nn.BatchNorm2d(64)
        self.bn5_1_2 = nn.BatchNorm2d(64)

        self.conv5_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn5_2 = nn.BatchNorm2d(64)
        self.conv5_3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn5_3 = nn.BatchNorm2d(64)


    def forward(self, BF, encoder2, encoder1=None, encoder3=None, Y=None):
        # encoder1: encoder n-1 layer
        # encoder2: encoder n   layer
        # encoder3: encoder n+1 layer
        # Y: Y n-1 block

        encoder2_size = encoder2.size()[2:]

        BF_re = F.interpolate(BF, size=encoder2_size, mode='bilinear')

        if encoder1 is None:
            encoder1_conv = torch.zeros_like(encoder2)
        else:
            encoder1_down = F.interpolate(encoder1, size=encoder2_size, mode='bilinear')
            encoder1_conv = self.conv1_1(encoder1_down)

        if encoder3 is None:
            encoder3_conv = torch.zeros_like(encoder2)
        else:
            encoder3_up = F.interpolate(encoder3, size=encoder2_size, mode='bilinear')
            encoder3_conv = self.conv1_3(encoder3_up)

        AfterAdd1_conv = F.relu(self.bn2(self.conv2(encoder2 + encoder1_conv + encoder3_conv)), inplace=True)

        AfterAdd2 = AfterAdd1_conv + encoder2

        if Y is None:
            AfterCat_conv1 = F.relu(self.bn5_1_2(self.conv5_1_2(torch.cat([AfterAdd2, BF_re], dim=1))), inplace=True)
        else:
            Y_conv = self.conv4(F.interpolate(Y, size=encoder2_size, mode='bilinear'))
            AfterCat_conv1 = F.relu(self.bn5_1_1(self.conv5_1_1(torch.cat([AfterAdd2, Y_conv, BF_re], dim=1))), inplace=True)

        AfterCat_conv2 = F.relu(self.bn5_2(self.conv5_2(AfterCat_conv1)), inplace=True)
        AfterCat_conv3 = F.relu(self.bn5_3(self.conv5_3(AfterCat_conv2)), inplace=True)

        out = AfterCat_conv3
        return out

# IFF
class Y_block(nn.Module):
    def __init__(self):
        super(Y_block, self).__init__()

        # for body
        self.conv1_b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1_b = nn.BatchNorm2d(64)

        self.conv2_b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2_b = nn.BatchNorm2d(64)

        # for detail
        self.conv1_d = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1_d = nn.BatchNorm2d(64)

        self.conv2_d = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2_d = nn.BatchNorm2d(64)


        # after concat
        self.conv3_1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3_1 = nn.BatchNorm2d(64)
        self.conv3_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3_2 = nn.BatchNorm2d(64)
        self.conv3_3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3_3 = nn.BatchNorm2d(64)

        self.fc = nn.Linear(in_features=128, out_features=64)


    def forward(self, body_map, detail_map):

        body_map_size = body_map.size()[2:]
        body_conv1   = F.relu(self.bn1_b(self.conv1_b(body_map)), inplace=True)
        detail_conv1 = F.relu(self.bn1_d(self.conv1_d(detail_map)), inplace=True)

        AfterAdd_body   = body_map + detail_conv1
        AfterAdd_detail = body_conv1 + detail_map

        body_conv2   = F.relu(self.bn2_b(self.conv2_b(AfterAdd_body)), inplace=True)
        detail_conv2 = F.relu(self.bn2_d(self.conv2_d(AfterAdd_detail)), inplace=True)

        AfterAdd_BD = body_conv2 + detail_conv2

        AfterAVE = AfterAdd_BD.mean(-1).mean(-1)
        AfterMAX = AfterAdd_BD.max(-1)[0].max(-1)[0]

        AfterCat_MaxAve = torch.cat([AfterAVE, AfterMAX], dim=1)

        AfterFC = self.fc(AfterCat_MaxAve)
        AfterFC_unsqueeze = AfterFC.unsqueeze(-1).unsqueeze(-1)

        AfterMul_conv1 = F.relu(self.bn3_1(self.conv3_1(AfterFC_unsqueeze * AfterAdd_BD)), inplace=True)
        AfterMul_conv2 = F.relu(self.bn3_2(self.conv3_2(AfterMul_conv1)), inplace=True)
        AfterMul_conv3 = F.relu(self.bn3_3(self.conv3_3(AfterMul_conv2)), inplace=True)


        out = AfterMul_conv3

        return out



# DINet
class XY(nn.Module):
    def __init__(self, cfg, imagenet_pretrained=True):
        super(XY, self).__init__()
        # b:1  d:2
        self.cfg = cfg
        self.bkbone = res2net50_v1b_26w_4s(pretrained=imagenet_pretrained)

        self.conv5b = nn.Sequential(nn.Conv2d(2048, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv4b = nn.Sequential(nn.Conv2d(1024, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv3b = nn.Sequential(nn.Conv2d(512, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv2b = nn.Sequential(nn.Conv2d(256, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv1b = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.conv5d = nn.Sequential(nn.Conv2d(2048, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv4d = nn.Sequential(nn.Conv2d(1024, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv3d = nn.Sequential(nn.Conv2d(512, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv2d = nn.Sequential(nn.Conv2d(256, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv1d = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.convBF = nn.Sequential(nn.Conv2d(320, 64, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.X1_1 = Xb_block()
        self.X1_2 = Xb_block()
        self.X1_3 = Xb_block()
        self.X1_4 = Xb_block()
        self.X1_5 = Xb_block()

        self.X2_1 = Xd_block()
        self.X2_2 = Xd_block()
        self.X2_3 = Xd_block()
        self.X2_4 = Xd_block()
        self.X2_5 = Xd_block()

        self.Y1 = Y_block()
        self.Y2 = Y_block()
        self.Y3 = Y_block()
        self.Y4 = Y_block()
        self.Y5 = Y_block()

        self.linearY5 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.linearY4 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.linearY3 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.linearY2 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.linearY1 = nn.Conv2d(64, 1, kernel_size=3, padding=1)

        self.linearB5 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.linearB4 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.linearB3 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.linearB2 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.linearB1 = nn.Conv2d(64, 1, kernel_size=3, padding=1)

        self.linearD5 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.linearD4 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.linearD3 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.linearD2 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.linearD1 = nn.Conv2d(64, 1, kernel_size=3, padding=1)

        self.initialize()

    def forward(self, x, shape=None):

        x = x.type(torch.cuda.FloatTensor)
        # print("x.size:", x.size())
        out1 = self.bkbone.conv1(x)
        out1 = self.bkbone.bn1(out1)
        out1 = self.bkbone.relu(out1)
        # print("out1.size:", out1.size())
        out2 = self.bkbone.maxpool(out1)  # bs, 64
        out2 = self.bkbone.layer1(out2)  # bs, 256
        # print("out2.size:", out2.size())
        out3 = self.bkbone.layer2(out2)  # bs, 512
        # print("out3.size:", out3.size())
        out4 = self.bkbone.layer3(out3)  # bs, 1024
        # print("out4.size:", out4.size())
        out5 = self.bkbone.layer4(out4)  # bs, 2048
        # print("out5.size:", out5.size())
        out1b, out1d = self.conv1b(out1), self.conv1d(out1)
        out2b, out3b, out4b, out5b   = self.conv2b(out2), self.conv3b(out3), self.conv4b(out4), self.conv5b(out5)
        out2d, out3d, out4d, out5d   = self.conv2d(out2), self.conv3d(out3), self.conv4d(out4), self.conv5d(out5)

        min_shape = out5.size()[2:]
        out1b_BF = F.interpolate(out1b, size=min_shape)
        out2b_BF = F.interpolate(out2b, size=min_shape)
        out3b_BF = F.interpolate(out3b, size=min_shape)
        out4b_BF = F.interpolate(out4b, size=min_shape)
        out5b_BF = F.interpolate(out5b, size=min_shape)
        BF_cat = torch.cat([out1b_BF, out2b_BF, out3b_BF, out4b_BF, out5b_BF], dim=1)
        BF = self.convBF(BF_cat)

        if shape is None:
            shape = x.size()[2:]

        # print("shape:", shape)
        Xb_5 = self.X1_5(BF=BF, encoder1=out4b, encoder2=out5b)
        Xd_5 = self.X2_5(encoder1=out4d, encoder2=out5d)
        Y5_out = self.Y5(Xb_5, Xd_5)

        Out_B5 = F.interpolate(self.linearB5(Xb_5), size=shape, mode='bilinear')
        Out_D5 = F.interpolate(self.linearD5(Xd_5), size=shape, mode='bilinear')
        Out_Y5 = F.interpolate(self.linearY5(Y5_out), size=shape, mode='bilinear')
        # print("Y5_out.size:", Y5_out.size())

        Xb_4 = self.X1_4(BF=BF, encoder1=out3b, encoder2=out4b, encoder3=out5b, Y=Y5_out)
        Xd_4 = self.X2_4(encoder1=out3d, encoder2=out4d, encoder3=out5d, Y=Y5_out)
        Y4_out = self.Y4(Xb_4, Xd_4)

        Out_B4 = F.interpolate(self.linearB4(Xb_4), size=shape, mode='bilinear')
        Out_D4 = F.interpolate(self.linearD4(Xd_4), size=shape, mode='bilinear')
        Out_Y4 = F.interpolate(self.linearY4(Y4_out), size=shape, mode='bilinear')
        # print("Y4_out.size:", Y4_out.size())

        Xb_3 = self.X1_3(BF=BF, encoder1=out2b, encoder2=out3b, encoder3=out4b, Y=Y4_out)
        Xd_3 = self.X2_3(encoder1=out2d, encoder2=out3d, encoder3=out4d, Y=Y4_out)
        Y3_out = self.Y3(Xb_3, Xd_3)

        Out_B3 = F.interpolate(self.linearB3(Xb_3), size=shape, mode='bilinear')
        Out_D3 = F.interpolate(self.linearD3(Xd_3), size=shape, mode='bilinear')
        Out_Y3 = F.interpolate(self.linearY3(Y3_out), size=shape, mode='bilinear')
        # print("Y3_out.size:", Y3_out.size())

        Xb_2 = self.X1_2(BF=BF, encoder1=out1b, encoder2=out2b, encoder3=out3b, Y=Y3_out)
        Xd_2 = self.X2_2(encoder1=out1d, encoder2=out2d, encoder3=out3d, Y=Y3_out)
        Y2_out = self.Y2(Xb_2, Xd_2)

        Out_B2 = F.interpolate(self.linearB2(Xb_2), size=shape, mode='bilinear')
        Out_D2 = F.interpolate(self.linearD2(Xd_2), size=shape, mode='bilinear')
        Out_Y2 = F.interpolate(self.linearY2(Y2_out), size=shape, mode='bilinear')
        # print("Y2_out.size:", Y2_out.size())

        Xb_1 = self.X1_1(BF=BF, encoder2=out1b, encoder3=out2b, Y=Y2_out)
        Xd_1 = self.X2_1(encoder2=out1d, encoder3=out2d, Y=Y2_out)
        Y1_out = self.Y1(Xb_1, Xd_1)

        Out_Y1 = F.interpolate(self.linearY1(Y1_out), size=shape, mode='bilinear')
        Out_B1 = F.interpolate(self.linearB1(Xb_1), size=shape, mode='bilinear')
        Out_D1 = F.interpolate(self.linearD1(Xd_1), size=shape, mode='bilinear')
        # print("Y1_out.size:", Y1_out.size())

        return Out_Y5, Out_Y4, Out_Y3, Out_Y2, Out_Y1, Out_B5, Out_B4, Out_B3, Out_B2, Out_B1, Out_D5, Out_D4, Out_D3, Out_D2, Out_D1

    def initialize(self):
        if self.cfg.snapshot:
            self.load_state_dict(torch.load(self.cfg.snapshot))