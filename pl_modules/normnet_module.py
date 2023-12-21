# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import pytorch_lightning as pl
import torch.nn as nn
from torchmetrics import R2Score, MeanAbsoluteError
import torchvision
import torch
import torchvision.transforms as T

import sys
from pathlib import Path

ROOT = str(Path(__file__).parent.parent)
sys.path.append(ROOT)

def l1_loss(x, y, f=lambda x:x):
    return ((f(x) - f(y)).abs()).mean()

class NormNetModule(pl.LightningModule):
    def __init__(self, backbone_cls,
                 opt = None, 
                 sched = None, 
                 loss = l1_loss, 
                 **kwargs):
        super().__init__()
        
        self.__dict__.update(**locals())
        self.cls = self.__class__
        self.save_hyperparameters(ignore=["self"])

        self.backbone = backbone_cls(n_classes=6)
        
        self.l1 = nn.ModuleDict(dict(_train=MeanAbsoluteError(compute_on_step=False),
                                     _val=MeanAbsoluteError(compute_on_step=False)))
        
    def step(self, batch, batch_nb, domain='train'):
        pred = self.backbone(batch['img'])   
        loss = self.loss(pred, batch['percs'])
        log_args = dict(sync_dist = (domain !='train'))
        self.log(f'{domain}_loss', loss, **log_args)
        return loss
    
    def epoch_end(self, outputs, domain='train'):
        l1 = self.l1['_'+domain].compute()
        self.log(f'{domain}_l1', l1, sync_dist=(domain !='train'))
        
    def training_step(self, batch, batch_nb):
        return self.step(batch, batch_nb, domain='train')
    
    def validation_step(self, batch, batch_nb):
        return self.step(batch, batch_nb, domain='val')
    
    def training_epoch_end(self, outputs):
        self.epoch_end(outputs, domain='train')
        
    def validation_epoch_end(self, outputs):
        self.epoch_end(outputs, domain='val')
          
    def configure_optimizers(self):
        opt = self.opt(self.parameters())
        sched = self.sched(opt)
        return [opt], [sched]