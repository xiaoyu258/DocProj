import numpy as np
from skimage import transform as tf
from skimage import io
import os
import skimage
import shutil
import argparse

import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
# For parsing commandline arguments
parser.add_argument("--res", type=int, default=256, help='the resolution of local patch')
parser.add_argument("--resFactor", type=int, default=4, help='the resolution factor of the global patch to the local patch')
parser.add_argument("--totalNum", type=int, default=2, help='the number of images to be cropped for the whole dataset')
parser.add_argument("--testNum", type=int, default=1, help='the number of images to be cropped for test dataset')
parser.add_argument("--dataPath", type=str, default='/home/xliea/SampleDataset', help='dataset path')
parser.add_argument("--savePath", type=str, default='/home/xliea/dataset_patch', help='save path')
args = parser.parse_args()

def createDir(savePath, resFactor):
        
    path =  '%s%s%s%s' % (savePath, '/train/patch_', resFactor, 'x')
    if not os.path.exists(path):
        os.makedirs(path)
        
    path =  '%s%s%s%s' % (savePath, '/test/patch_', resFactor, 'x')
    if not os.path.exists(path):
        os.makedirs(path)

        
def moveTest(num, testNum, savePath, resFactor, datasetpath):
    
    testDir = os.listdir('%s%s' % (datasetpath, '/img'))[(num - testNum):num]
    testDir.sort()
    
    imgPaths = os.listdir('%s%s%s%s' % (savePath, '/train/patch_', resFactor, 'x'))
    imgPaths.sort()
    
    for fs in imgPaths:
        
        testIndex = '%s%s' % (fs[0:5], '.png')
        
        if testIndex in testDir:

            name = fs.split('.')[0]

            dir1 = '%s%s%s%s%s%s' % (savePath, '/train/patch_', resFactor, 'x/', name, '.png')
            dir2 = '%s%s%s%s%s%s' % (savePath, '/test/patch_', resFactor, 'x/', name, '.png')
            shutil.move(dir1, dir2)

def generate(S, num, savePath, resFactor, datasetpath):
    
    createDir(savePath, resFactor)
    
    ind = 0

    for fs in os.listdir('%s%s' % (datasetpath, '/img'))[0:num]:

        name = fs.split('.')[0]
        
        print(ind, name)
        ind += 1

        imgpath =  '%s%s%s%s' % (datasetpath, '/img/', name, '.png')
        mskpath =  '%s%s%s%s' % (datasetpath, '/img_msk/', name, '.png')
        
        image = io.imread(imgpath)
        mask = io.imread(mskpath)

        H = image.shape[0]
        W = image.shape[1]
        
        cropH = S
        cropW = S
        ovlp = int(S * 0.25)
        
        ynum = int((H - cropH)/(cropH - ovlp)) + 1
        xnum = int((W - cropW)/(cropW - ovlp)) + 1
        
        padding = int(S * (resFactor - 1.0)/2.0)
        
        image = np.lib.pad(image, ((padding, padding), (padding, padding), (0, 0)), 'edge')

        for j in range(0, ynum):
            for i in range(0, xnum):

                gx = int(i * (cropW - ovlp))
                gy = int(j * (cropH - ovlp))
                
                patchmask = mask[gy:int(gy + S), gx:int(gx + S)]
                globalpatch = image[gy:int(gy + resFactor*S), gx:int(gx + resFactor*S)]
                
                if np.sum(patchmask) == 0:

                    newimg = skimage.transform.resize(globalpatch, (256, 256))

                    imgpath =  '%s%s%s%s%s%s%s%s%s' % (savePath, '/train/patch_', resFactor, 'x/', name, '_', 
                                                       str(j).zfill(2), str(i).zfill(2), '.png')

                    io.imsave(imgpath, newimg) 

generate(args.res, args.totalNum, args.savePath, args.resFactor, args.dataPath)
moveTest(args.totalNum, args.testNum, args.savePath, args.resFactor, args.dataPath)
