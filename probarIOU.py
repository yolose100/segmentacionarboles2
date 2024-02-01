#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 13:34:59 2019

@author: etellez
"""

#!/usr/bin/env python
"""Test UNet and create a Kaggle submission."""
__author__ = 'Erdene-Ochir Tuguldur, Yuan Xu'

import time
import argparse
from tqdm import tqdm
import cv2

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
import models
from torchvision.transforms import *
from skimage.io import imread
from datasets import *
from transforms import *
import skimage 
from utils import rlenc, rlenc_np, FasterRle, gzip_save
import scipy.misc
import imageio
from PIL import Image
import numpy
from skimage.filters import threshold_otsu
from utils.metrics import calc_metric
import os.path as path
orig_img_size = 256
img_size = 256
padding = compute_padding(orig_img_size, orig_img_size, img_size)
d_y0, d_y1, d_x0, d_x1 = padding
y0, y1, x0, x1 = d_y0, d_y0 + orig_img_size, d_x0, d_x0 + orig_img_size


def predict(model, batch,  use_gpu):
    image_ids, inputs = batch['image_id'], batch['input']
    image_ids, inputs, targets = batch['image_id'], batch['input'], batch['mask']
    #logit, logit_pixel, logit_image = model(inputs)

    if use_gpu:
        inputs = inputs.cuda()
    print (inputs.dtype)
    outputs, _, _ = model(inputs)
    print("salvar a archivos demora bastante ")
    indiceBatch = 0  ;
    mascaras=[]
           
    targets_numpy = targets.cpu().numpy()
    squeeze = torch.squeeze(outputs)
    probs_numpy =squeeze .cpu().detach().numpy()
    threshold  = threshold_otsu(probs_numpy)
    predictions_numpy = probs_numpy >0.5  # predictions.cpu().numpy()
    metric_array = calc_metric(targets_numpy, predictions_numpy, type='iou', size_average=False)
    
    metric_array2 = calc_metric(targets_numpy, predictions_numpy, type='pixel_accuracy', size_average=False)

    metric = metric_array2.mean()

    print(str(metric))
    for  i in image_ids:
        temp = outputs[indiceBatch][0].numpy()
        temp = temp.astype(numpy.float32)
        s = threshold_otsu(temp)
        probs =temp>=0.5
        probs =probs.astype(numpy.float32)
        print (str(probs))
        print(str(outputs.shape))
        imageio.imwrite('outfile'+str(i)+'.png', temp)
        imageio.imwrite('outfileSi'+str(i)+'.png', probs)
        indiceBatch = indiceBatch+1
        mascaras.append(probs)
    return outputs,mascaras,metric_array2

 

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
        out,mascaras,metric_array=predict(model, batch, use_gpu=use_gpu)
        return out,mascaras,metric_array
def mostrarInfo(lista, idImagen,out,mascaras,metric):
    matplotlib.rcParams['font.size'] = 9
    
    try:
        image = imread("./mytrain/train_"+str(lista[idImagen])+".jpg")
    except :
        image = imread("./mytrain/train_"+str(lista[idImagen])+".tif")

    archivo = "./mytrain/train_"+str(lista[idImagen])+"_json/label.png"
    if path.exists(archivo):
        mask = imread(archivo, as_gray=True)
    else:
        mask = image
   
    mask = cv2.resize(mask,(128,128))
    mask = mask >0
    fig, ax = plt.subplots(2, 2, figsize=(14, 13))
    ax1, ax2, ax3, ax4 = ax.ravel()
    
    ax1.imshow(image, cmap=plt.cm.gray)
    ax1.set_title('Original')
    ax1.axis('off')
    
    ax2.imshow(mascaras[idImagen], cmap=plt.cm.gray)
    ax2.set_title('resultado')
    ax2.axis('off')
    
    ax3.imshow(mask, cmap=plt.cm.gray)
    ax3.set_title('resultado Esperado')
    ax3.axis('off')
    
    
#    ax4.imshow(out[idImagen][0].numpy(), cmap=plt.cm.gray)
#    ax4.set_title('sin binarisar')
#    ax4.axis('off')
    
    comparacion =(mask +mascaras[idImagen])%2
    suma  =np.sum(comparacion)
    resultado = 1.0-suma /(128.0*128.0)
    print (resultado)
    ax4.imshow(comparacion, cmap=plt.cm.gray)
    ax4.set_title('sin binarisar')
    ax4.axis('off')
    
    plt.show()


if __name__ == '__main__':
   

    lista= [51371,51373,51597,51606,51607,51608,51609,51610]
    use_gpu = False
    print('use_gpu', use_gpu)

    print("loading model...")
    saved_checkpoint = torch.load("runs/fold2/checkpoints/last-checkpoint-fold2.pth")
    old_model = models.load(saved_checkpoint['model_file'])
    old_model = old_model.cpu()
    model = old_model.float()   
    since = time.time()
    out,mascaras,metric=test(lista)
    rx = 0 
    for i in range(len(lista)):
        print ("**********************************")
        print (str(rx))
        mostrarInfo(lista,i,out,mascaras,metric)
        rx = rx+1
    

    time_elapsed = time.time() - since
    time_str = 'total time elapsed: {:.0f}h {:.0f}m {:.0f}s '.format(time_elapsed // 3600, time_elapsed % 3600 // 60,
                                                                     time_elapsed % 60)
    print("finished")
