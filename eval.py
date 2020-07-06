import os
import cv2
import shutil
import numpy as np
import skimage
from skimage import io
from skimage import transform as tf
import argparse

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from collections import OrderedDict

from modelGeoNet import GeoNet

# For parsing commandline arguments
parser = argparse.ArgumentParser()
parser.add_argument("--imgPath", type=str, default='E:\\IMG_7325.jpeg', help='input image path')
parser.add_argument("--modelPath", type=str, default='E:\\model.pkl', help='pre-trained model path')
parser.add_argument("--saveImgPath", type=str, default='E:\\IMG_7325.png', help='resized image path')
parser.add_argument("--saveFlowPath", type=str, default='E:\\IMG_7325.npy', help='saved flows path')
args = parser.parse_args()

def resizeImg(imgPath, H, W):
    
    '''
    resize while keeping the aspect ratio and then crop the image to a given shape (H, W)
    '''

    img = io.imread(imgPath)
    h, w = img.shape[0:2]
    
    if h > w:
        ratio = float(h)/float(w)

        if (ratio > float(H)/float(W)):
            img = skimage.transform.resize(img, (int(ratio*W), W), order=1)
        else:
            img = skimage.transform.resize(img, (H, int(H/ratio)), order=1)

        yc = int(img.shape[0]/2)
        xc = int(img.shape[1]/2)
        img = img[yc - int(H/2):yc + int(H/2), xc - int(W/2):xc + int(W/2)]
        
    else:
        ratio = float(w)/float(h)
        
        if (ratio > float(H)/float(W)):
            img = skimage.transform.resize(img, (W, int(W*ratio)), order=1)
        else:
            img = skimage.transform.resize(img, (int(H/ratio), H), order=1)
         
        yc = int(img.shape[0]/2)
        xc = int(img.shape[1]/2)
        img = img[yc - int(W/2):yc + int(W/2), xc - int(H/2):xc + int(H/2)]
        
    return img

def padImg(img):
    '''
    pad image twice.
    The first padding is to make sure the patches cover all image regions.
    The second padding is used for cropping the global patch.
    '''
    
    H = img.shape[0]
    W = img.shape[1]
    
    globalFct = 4
    patchRes = 256
    ovlp = int(patchRes * 0.25)
    
    padH = (int((H - patchRes)/(patchRes - ovlp) + 1) * (patchRes - ovlp) + patchRes) - H
    padW = (int((W - patchRes)/(patchRes - ovlp) + 1) * (patchRes - ovlp) + patchRes) - W
    
    padding = int(patchRes * (globalFct - 1) / 2.0)

    padImg = cv2.copyMakeBorder(img, 0, padH, 0, padW, cv2.BORDER_REPLICATE)
    padImg = cv2.copyMakeBorder(padImg, padding, padding, padding, padding, cv2.BORDER_REPLICATE)
    
    return padImg

def cropToPatch(img):
    '''
    crop the image to local and global patches
    '''

    H = img.shape[0]
    W = img.shape[1]
    
    globalFct = 4
    patchRes = 256
    ovlp = int(patchRes * 0.25)
    padding = int(patchRes * (globalFct - 1) / 2.0)

    cropH = patchRes
    cropW = patchRes

    ynum = int((H - (globalFct - 1) * cropH - cropH)/(cropH - ovlp)) + 1
    xnum = int((W - (globalFct - 1) * cropW - cropW)/(cropW - ovlp)) + 1
    
    totalLocal = np.zeros((ynum, xnum, patchRes, patchRes, 3), dtype=np.uint8)
    totalGloba = np.zeros((ynum, xnum, 256, 256, 3), dtype=np.uint8)

    for j in range(0, ynum):
        for i in range(0, xnum):

            x = int(padding + i * (cropW - ovlp))
            y = int(padding + j * (cropH - ovlp))

            totalLocal[j, i] = img[y:int(y + patchRes), x:int(x + patchRes)]

            gx = int(x - padding)
            gy = int(y - padding)
            globalpatch = img[gy:int(gy + globalFct * patchRes), gx:int(gx + globalFct * patchRes)]
            globalpatch = skimage.transform.resize(globalpatch, (256, 256)) * 255.0
            totalGloba[j, i] = globalpatch
            
    return totalLocal, totalGloba



def testRealFlow(modelPath, localPatch, globalPatch):
    '''
    estimate the flows
    '''

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    model = GeoNet([1, 1, 1, 1, 1])

    if torch.cuda.is_available():
        model = model.cuda()
        
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        model.load_state_dict(torch.load(modelPath))
    else:
        state_dict = torch.load(modelPath)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)  
        
    model.eval()
    
    ynum = localPatch.shape[0]
    xnum = localPatch.shape[1]
    scal = localPatch.shape[2]
    
    totalFlow = np.zeros((ynum, xnum, 2, scal, scal), dtype = np.float32)
    
    for j in range(0, ynum):
        for i in range(0, xnum):

            temp_localPatch = localPatch[j, i]
            temp_globaPatch = globalPatch[j, i]
        
            temp_localPatch = transform(temp_localPatch)
            temp_globaPatch = transform(temp_globaPatch)

            if torch.cuda.is_available():
                temp_localPatch = temp_localPatch.cuda()
                temp_globaPatch = temp_globaPatch.cuda()

            temp_localPatch = temp_localPatch.view(1,3,scal,scal)
            temp_globaPatch = temp_globaPatch.view(1,3,256,256)
            
            temp_localPatch = Variable(temp_localPatch)
            temp_globaPatch = Variable(temp_globaPatch)
            
            flow_output = model(temp_localPatch, temp_globaPatch)

            u = flow_output.data.cpu().numpy()[0][0]
            v = flow_output.data.cpu().numpy()[0][1]
            
            totalFlow[j,i,0] = u
            totalFlow[j,i,1] = v

    return totalFlow


img = resizeImg(args.imgPath, H = 2000, W = 1500)
io.imsave(args.saveImgPath, img)
img = padImg(img)
totalLocalPatch, totalGlobaPatch = cropToPatch(img)
totalFlow = testRealFlow(args.modelPath, totalLocalPatch, totalGlobaPatch)
np.save(args.saveFlowPath, totalFlow)
