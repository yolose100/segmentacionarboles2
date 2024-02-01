#!/usr/bin/env python
"""Test UNet and create a Kaggle submission."""
__author__ = 'Erdene-Ochir Tuguldur, Yuan Xu'

import time
import argparse
from tqdm import tqdm

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
import models
from torchvision.transforms import *

from datasets import *
from transforms import *
import skimage 
from utils import rlenc, rlenc_np, FasterRle, gzip_save
import scipy.misc
import imageio
from PIL import Image
import numpy
from skimage.filters import threshold_otsu

orig_img_size = 256
img_size = 128
padding = compute_padding(orig_img_size, orig_img_size, img_size)
d_y0, d_y1, d_x0, d_x1 = padding
y0, y1, x0, x1 = d_y0, d_y0 + orig_img_size, d_x0, d_x0 + orig_img_size


def predict(model, batch,  use_gpu):
    image_ids, inputs = batch['image_id'], batch['input']
    if use_gpu:
        inputs = inputs.cuda()
    print (inputs.dtype)
    outputs, _, _ = model(inputs)
    print("salvar a archivos demora bastante ")
    indiceBatch = 0  ;
    mascaras=[]
    for  i in image_ids:
        temp = outputs[indiceBatch][0].numpy()
        temp = temp.astype(numpy.float32)
        s = threshold_otsu(temp)
        probs =temp>s
        probs =probs.astype(numpy.float32)
        print (str(probs))
        print(str(outputs.shape))
        imageio.imwrite('outfile_'+str(i.item())+'.png', temp)
        imageio.imwrite('outfileSi_'+str(i.item())+'.png', probs)
        indiceBatch = indiceBatch+1
        mascaras.append(probs)
    return outputs,mascaras

 

def test(arr):
    test_transform = Compose([PrepareImageAndMask(),
                              HWCtoCHW()])
    arrParaTest=arr
    test_dataset = SaltIdentification(mode='test', transform=test_transform, preload=False ,pruebas = arrParaTest)
    test_dataloader = DataLoader(test_dataset, batch_size=len(arr))

 #   model.eval()
    torch.set_grad_enabled(False)

    pbar = tqdm(test_dataloader, unit="images", unit_scale=test_dataloader.batch_size, disable=None)

    for batch in pbar:
        print("al momento de enviar"+str(batch['input'].dtype))
        out,mascaras=predict(model, batch, use_gpu=use_gpu)
        return out,mascaras

if __name__ == '__main__':
   
    use_gpu = False
    print('use_gpu', use_gpu)

    print("loading model...")
#    model = models.load("mejorMetrica.pth")
#:    model.float()
    saved_checkpoint = torch.load("runs/fold2/checkpoints/last-checkpoint-fold2.pth")
    old_model = models.load(saved_checkpoint['model_file'])
    old_model = old_model.cpu()
    model = old_model.float()

    since = time.time()
    out,mascaras=test([1,141])
    
    
    
    matplotlib.rcParams['font.size'] = 9
    
    resultado = (mascaras[0]+mascaras[1])%2

    imageio.imwrite('outfile_resta.png', resultado)
    
    fig, ax = plt.subplots(2, 2, figsize=(8, 5))
    ax1, ax2, ax3, ax4 = ax.ravel()
    
    ax1.imshow(mascaras[0], cmap=plt.cm.gray)
    ax1.set_title('Original 1')
    ax1.axis('off')
    
    ax2.imshow(mascaras[1], cmap=plt.cm.gray)
    ax2.set_title('Original 2 ')
    ax2.axis('off')
    
    ax3.imshow(resultado, cmap=plt.cm.gray)
    ax3.set_title('resta')
    ax3.axis('off')
    
    plt.show()
    time_elapsed = time.time() - since
    time_str = 'total time elapsed: {:.0f}h {:.0f}m {:.0f}s '.format(time_elapsed // 3600, time_elapsed % 3600 // 60,
                                                                     time_elapsed % 60)
    print("finished")
