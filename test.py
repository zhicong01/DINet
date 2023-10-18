#!/usr/bin/python3
#coding=utf-8

import os
import sys
sys.path.insert(0, '../')
sys.dont_write_bytecode = True

import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

import torch
import dataset
from torch.utils.data import DataLoader
from net import XY

class Test(object):
    def __init__(self, Dataset, Network, Path, model_path, model_num):

        ## dataset
        self.model_num = model_num
        self.cfg    = Dataset.Config(datapath=Path, snapshot=model_path, mode='test')
        self.data   = Dataset.Data(self.cfg)
        self.loader = DataLoader(self.data, batch_size=1, shuffle=False, num_workers=8)
        ## network
        self.net    = Network(self.cfg)
        self.net.train(False)
        self.net.cuda()

    def save(self):
        with torch.no_grad():
            for image, (H, W), name in self.loader:
                image, shape  = image.cuda().float(), (H, W)
                # outb1, outd1, out1, outb2, outd2, out2 = self.net(image, shape)
                Out_Y5, Out_Y4, Out_Y3, Out_Y2, Out_Y1, Out_B5, Out_B4, Out_B3, Out_B2, Out_B1, Out_D5, Out_D4, Out_D3, Out_D2, Out_D1 = self.net(image, shape)
                
                # # wzc add
                # pred = torch.sigmoid(Out_B1[0, 0]).cpu().numpy() * 255
                # head = self.cfg.datapath + '/body'
                # if not os.path.exists(head):
                #     os.makedirs(head)
                # cv2.imwrite(head + '/' + name[0] + '.png', np.round(pred))
                #
                # pred = torch.sigmoid(Out_D1[0, 0]).cpu().numpy() * 255
                # head = self.cfg.datapath + '/detail'
                # if not os.path.exists(head):
                #     os.makedirs(head)
                # cv2.imwrite(head + '/' + name[0] + '.png', np.round(pred))
                # #########
                
                out  = Out_Y1
                pred = torch.sigmoid(out[0,0]).cpu().numpy()*255
                head = './eval/maps/v1.1.1epoch'+ str(self.model_num) + '/' + self.cfg.datapath.split('/')[-1]
                if not os.path.exists(head):
                    os.makedirs(head)
                cv2.imwrite(head+'/'+name[0]+'.png', np.round(pred))


if __name__=='__main__':
    start_model = 102
    stop_model = 120

    for i in range(start_model, stop_model + 1):
        for path in ['/home/wuzhicong/COD/LDF-master/data/CAMO', '/home/wuzhicong/COD/LDF-master/data/CHAMELEON',
                     '/home/wuzhicong/COD/LDF-master/data/COD10K', '/home/wuzhicong/COD/LDF-master/data/NC4K']:
            model_path = './out/model-%d' % i
            t = Test(dataset, XY, path, model_path, i)
            t.save()
