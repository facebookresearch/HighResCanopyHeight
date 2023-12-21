# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import argparse
import os
import torch
import pandas as pd
import numpy as np
import torchvision.transforms as T
import matplotlib.pyplot as plt
import torchmetrics
from pathlib import Path
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
import math
import torchvision.transforms.functional as TF
import torchvision
from torchvision.utils import save_image

from models.backbone import SSLVisionTransformer
from models.dpt_head import DPTHead
import pytorch_lightning as pl
from models.regressor import RNet

class SSLAE(nn.Module):
    def __init__(self, pretrained=None, classify=True, n_bins=256, huge=False):
        super().__init__()
        if huge == True:
            self.backbone = SSLVisionTransformer(
            embed_dim=1280,
            num_heads=20,
            out_indices=(9, 16, 22, 29),
            depth=32,
            pretrained=pretrained
            )
            self.decode_head = DPTHead(
                classify=classify,
                in_channels=(1280, 1280, 1280, 1280),
                embed_dims=1280,
                post_process_channels=[160, 320, 640, 1280],
            )  
        else:
            self.backbone = SSLVisionTransformer(pretrained=pretrained)
            self.decode_head = DPTHead(classify=classify,n_bins=256)
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.decode_head(x) 
        return x

class SSLModule(pl.LightningModule):
    def __init__(self, 
                  ssl_path="compressed_SSLbaseline.pth"):
        super().__init__()
    
        if 'huge' in ssl_path:
            self.chm_module_ = SSLAE(classify=True, huge=True).eval()
        else:
            self.chm_module_ = SSLAE(classify=True, huge=False).eval()
        
        if 'compressed' in ssl_path:   
            ckpt = torch.load(ssl_path, map_location='cpu')
            self.chm_module_ = torch.quantization.quantize_dynamic(
                self.chm_module_, 
                {torch.nn.Linear,torch.nn.Conv2d,  torch.nn.ConvTranspose2d},
                dtype=torch.qint8)
            self.chm_module_.load_state_dict(ckpt, strict=False)
        else:
            ckpt = torch.load(ssl_path)
            state_dict = ckpt['state_dict']
            self.chm_module_.load_state_dict(state_dict)
        
        self.chm_module = lambda x: 10*self.chm_module_(x)
    def forward(self, x):
        x = self.chm_module(x)
        return x

class NeonDataset(torch.utils.data.Dataset):
    path = './data/images/'
    root_dir = Path(path)
    df_path = './data/neon_test_data.csv'
    
    def __init__(self, model_norm, new_norm, src_img='maxar', 
                 trained_rgb= False, no_norm = False,
                **kwargs):
       
        self.no_norm = no_norm
        self.model_norm = model_norm
        self.new_norm = new_norm
        self.trained_rgb = trained_rgb
        self.size = 256
        self.df = pd.read_csv(self.df_path, index_col=0)
        self.src_img = src_img
        
        # number of times crops can be used horizontally
        self.size_multiplier = 6 
        
    def __len__(self):
        if self.src_img == 'neon':
            return 30 * len(self.df) 
        return len(self.df)
        

    def __getitem__(self, i):      
        n = self.size_multiplier 
        ix, jx, jy = i//(n**2), (i%(n**2))// n, (i% (n**2)) % n 
        if self.src_img == 'neon':
            l = self.df.iloc[ix]
        x = list(range(l.bord_x, l.imsize-l.bord_x-self.size, self.size))[jx]
        y = list(range(l.bord_y, l.imsize-l.bord_y-self.size, self.size))[jy]  
        img = TF.to_tensor(Image.open(self.root_dir / l[self.src_img]).crop((x, y, x+self.size, y+self.size)))
        chm = TF.to_tensor(Image.open(self.root_dir / l.chm).crop((x, y, x+self.size, y+self.size)))
        chm[chm<0] = 0
        
        if not self.trained_rgb:
            if self.src_img == 'neon':
                if self.no_norm:
                    normIn = img
                else:
                    if self.new_norm:
                        # image image normalization using learned quantiles of pairs of Maxar/Neon images
                        x = torch.unsqueeze(img, dim=0)
                        norm_img = self.model_norm(x).detach()
                        p5I = [norm_img[0][0].item(), norm_img[0][1].item(), norm_img[0][2].item()]
                        p95I = [norm_img[0][3].item(), norm_img[0][4].item(), norm_img[0][5].item()]
                    else:
                        # apply image normalization to aerial images, matching color intensity of maxar images
                        I = TF.to_tensor(Image.open(self.root_dir / l['maxar']).crop((x, y, x+s, y+s))) 
                        p5I = [np.percentile(I[i,:,:].flatten(),5) for i in range(3)]
                        p95I = [np.percentile(I[i,:,:].flatten(),95) for i in range(3)]
                    p5In = [np.percentile(img[i,:,:].flatten(),5) for i in range(3)]

                    p95In = [np.percentile(img[i,:,:].flatten(),95) for i in range(3)]
                    normIn = img.clone()
                    for i in range(3):
                        normIn[i,:,:] = (img[i,:,:]-p5In[i]) * ((p95I[i]-p5I[i])/(p95In[i]-p5In[i])) + p5I[i]
                  
        return {'img': normIn, 
                'img_no_norm': img, 
                'chm': chm,
                'lat':torch.Tensor([l.lat]).nan_to_num(0),
                'lon':torch.Tensor([l.lon]).nan_to_num(0),
               }

def evaluate(model, 
             norm, 
             model_norm,
             name, 
             bs=32, 
             trained_rgb=False,
             normtype=2,
             device = 'cuda:0', 
             no_norm = False, 
             display = False):
      
    dataset_key = 'neon_aerial'
    
    print("normtype", normtype)    
    
    # choice of the normalization of aerial images. 
    # i- For inference on satellite images args.normtype should be set to 0; 
    # ii- For inference on aerial images, if corresponding Maxar quantiles at the
    # same coordinates are known, args.normtype should be set to 1;
    # iii- For inference on aerial images, an automatic normalization using a pretrained
    # network on aerial and satellite images on Neon can be used: args.normtype should be set to 2 (default); 
    
    new_norm=True
    no_norm=False
    if normtype == 0:
        no_norm=True
    elif normtype == 1:
        new_norm=False
    elif normtype == 2:
        new_norm=True
    
    ds = NeonDataset( model_norm, new_norm, domain='test', src_img='neon', trained_rgb=trained_rgb, no_norm=no_norm)
    dataloader = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=True, num_workers=10)
        
    Path('../reports').joinpath(name).mkdir(parents=True, exist_ok=True)
    Path('../reports/'+name).joinpath('results_for_fig_'+dataset_key).mkdir(parents=True, exist_ok=True)
    metrics = {}

    # canopy height metrics
    metric_classes = dict(
        mae = torchmetrics.MeanAbsoluteError(),
        rmse = torchmetrics.MeanSquaredError(squared= False),
        r2 = torchmetrics.R2Score(),
        r2_block = torchmetrics.R2Score())
        
    downsampler = nn.AvgPool2d(50)
    bd = 3
    
    preds, chms = [], []
    chm_block_means, pred_block_means = [], []
    
    fig_batch_ind = 0

    for batch in tqdm(dataloader):
        chm = batch['chm'].detach()
        batch = {k:v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        pred = model(norm(batch['img']))
        pred = pred.cpu().detach().relu()
        
        if display == True:
            # display Predicted CHM
            for ind in range(pred.shape[0]):     
                fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(20, 5))
                plt.subplots_adjust(hspace=0.5)
                img_no_norm = batch['img_no_norm'][ind].cpu()
                Inn = np.moveaxis(img_no_norm.numpy(), 0, 2)
                img = batch['img'][ind].cpu()
                I = np.moveaxis(img.numpy(), 0, 2)
                gt = batch['chm'][ind].cpu()
                GT = np.moveaxis(gt.numpy(), 0, 2)
                ax[0].imshow(Inn)
                ax[0].set_title(f"Image",fontsize=12)
                ax[0].set_xlabel('meters')
                ax[1].imshow(I)
                ax[1].set_title(f"Normalized Image ",fontsize=12)
                ax[1].set_xlabel('meters')
                combined_data = np.concatenate((batch['chm'][ind].cpu().numpy(), pred[ind].detach().numpy()), axis=0)
                _min, _max = np.amin(combined_data), np.amax(combined_data)
                pltim = ax[2].imshow(pred[ind][0].detach().numpy(), vmin = _min, vmax = _max)
                ax[2].set_title(f"Pred CHM",fontsize=12)
                ax[2].set_xlabel('meters')
                pltim = ax[3].imshow(GT, vmin = _min, vmax = _max)
                ax[3].set_title(f"GT CHM",fontsize=12)
                ax[3].set_xlabel('meters') 
                cax = fig.add_axes([0.95, 0.15, 0.02, 0.7])
                fig.colorbar(pltim, cax=cax, orientation="vertical")
                cax.set_title("meters", fontsize=12) 
                plt.savefig(f"{name}/fig_{fig_batch_ind}_{ind}_{normtype}.png", dpi=300)
            
            fig_batch_ind = fig_batch_ind + 1
        
        chm_block_mean = downsampler(chm[..., bd:, bd:])
        pred_block_mean = downsampler(pred[..., bd:, bd:])
        
        metric_classes['mae'].update(pred, chm)
        metric_classes['rmse'].update(pred, chm)
        metric_classes['r2'].update(pred.flatten(), chm.flatten())
        metric_classes['r2_block'].update(pred_block_mean.flatten(), chm_block_mean.flatten())
    
        preds.append(pred), chms.append(chm)
        chm_block_means.append(chm_block_mean)
        pred_block_means.append(pred_block_mean)
        if display:
            break
    preds, chms = torch.cat(preds), torch.cat(chms)
    
    metrics = {k:v.compute() for k, v in metric_classes.items()}
    torch.save(metrics, f'{name}/metrics.pt')

    #print metrics
    for k, v in metrics.items():
        print(f'{k} {v.item():.2f}')
    print(f"Bias: {(preds.flatten()-chms.flatten()).numpy().mean():.2f}")
    

def parse_args():
    parser = argparse.ArgumentParser(
        description='test a model')
    parser.add_argument('--checkpoint', type=str, help='CHM pred checkpoint file', default='saved_checkpoints/compressed_SSLlarge.pth')
    parser.add_argument('--name', type=str, help='run name', default='output_inference')
    parser.add_argument('--trained_rgb', type=str, help='True if model was finetuned on aerial data')
    parser.add_argument('--normnet', type=str, help='path to a normalization network', default='saved_checkpoints/aerial_normalization_quantiles_predictor.ckpt')
    parser.add_argument('--normtype', type=int, help='0: no norm; 1: old norm, 2: new norm', default=2) 
    parser.add_argument('--display', type=bool, help='saving outputs in images')
    args = parser.parse_args()
    return args



def main():
    # 0- read args 
    args = parse_args()
    if 'compressed' in args.checkpoint:
        device='cpu'
    else:
        device='cuda:0'
    
    os.system("mkdir "+args.name)
    
    # 1- load network and its weight to normalize aerial images to match intensities from satellite images. 
    norm_path = args.normnet 
    ckpt = torch.load(norm_path, map_location='cpu')
    state_dict = ckpt['state_dict']
    for k in list(state_dict.keys()):
        if 'backbone.' in k:
            new_k = k.replace('backbone.','')
            state_dict[new_k] = state_dict.pop(k)
    
    model_norm = RNet(n_classes=6)
    model_norm = model_norm.eval()
    model_norm.load_state_dict(state_dict)
        
    # 2- load SSL model
    model = SSLModule(ssl_path = args.checkpoint)
    model.to(device)
    model = model.eval()
    
    # 3- image normalization for each image going through the encoder
    norm = T.Normalize((0.420, 0.411, 0.296), (0.213, 0.156, 0.143))
    norm = norm.to(device)
    
    # 4- evaluation 
    evaluate(model, norm, model_norm, name=args.name, bs=16, trained_rgb=args.trained_rgb, normtype=args.normtype, device=device, display=args.display)

if __name__ == '__main__':
    main()
