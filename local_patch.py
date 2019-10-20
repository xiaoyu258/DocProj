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
parser.add_argument("--totalNum", type=int, default=2, help='the number of images to be cropped for the whole dataset')
parser.add_argument("--testNum", type=int, default=1, help='the number of images to be cropped for test dataset')
parser.add_argument("--dataPath", type=str, default='/home/xliea/SampleDataset', help='dataset path')
parser.add_argument("--savePath", type=str, default='/home/xliea/dataset_patch', help='save path')
args = parser.parse_args()

def createDir(savePath):
    
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    
    path =  '%s%s' % (savePath, '/train/patch')
    if not os.path.exists(path):
        os.makedirs(path)
        
    path =  '%s%s' % (savePath, '/train/patch_flow')
    if not os.path.exists(path):
        os.makedirs(path)

    path =  '%s%s' % (savePath, '/test/patch')
    if not os.path.exists(path):
        os.makedirs(path)
        
    path =  '%s%s' % (savePath, '/test/patch_flow')
    if not os.path.exists(path):
        os.makedirs(path)
        
def moveTest(num, testNum, savePath, datasetpath):
    
    testDir = os.listdir('%s%s' % (datasetpath, '/img'))[(num - testNum):num]
    testDir.sort()
    
    imgPaths = os.listdir('%s%s' % (savePath, '/train/patch'))
    imgPaths.sort()
    
    for fs in imgPaths:
        
        testIndex = '%s%s' % (fs[0:5], '.png')
        
        if testIndex in testDir:

            name = fs.split('.')[0]

            dir1 = '%s%s%s%s' % (savePath, '/train/patch/', name, '.png')
            dir2 = '%s%s%s%s' % (savePath, '/test/patch/', name, '.png')
            shutil.move(dir1, dir2)

            dir5 = '%s%s%s%s' % (savePath, '/train/patch_flow/', name, '.npy')
            dir6 = '%s%s%s%s' % (savePath, '/test/patch_flow/', name, '.npy')
            shutil.move(dir5, dir6)

def generate(S, num, savePath, datasetpath):
    
    createDir(savePath)
    
    ind = 0

    for fs in os.listdir('%s%s' % (datasetpath, '/img'))[0:num]:

        name = fs.split('.')[0]
        
        print(ind, name)
        ind += 1

        imgpath =  '%s%s%s%s' % (datasetpath, '/img/', name, '.png')
        mskpath =  '%s%s%s%s' % (datasetpath, '/img_msk/', name, '.png')
        matpath =  '%s%s%s%s' % (datasetpath, '/flow/', name, '.npy')

        image = io.imread(imgpath)
        mask = io.imread(mskpath)
        flow  = np.load(matpath)

        H = image.shape[0]
        W = image.shape[1]
        
        cropH = S
        cropW = S
        ovlp = int(S * 0.25)
        
        ynum = int((H - cropH)/(cropH - ovlp)) + 1
        xnum = int((W - cropW)/(cropW - ovlp)) + 1

        for j in range(0, ynum):
            for i in range(0, xnum):

                x = int(i * (cropW - ovlp))
                y = int(j * (cropH - ovlp))

                patch = image[y:int(y + S), x:int(x + S)]
                patchmask = mask[y:int(y + S), x:int(x + S)]
                patchFlowX = flow[0, y:int(y + S), x:int(x + S)]
                patchFlowY = flow[1, y:int(y + S), x:int(x + S)]

                if (S % 2 == 0):
                    cS = int(S/2)
                    xDis = np.mean(patchFlowX[(cS - 1):(cS + 1), (cS - 1):(cS + 1)])
                    yDis = np.mean(patchFlowY[(cS - 1):(cS + 1), (cS - 1):(cS + 1)])
                else:
                    cS = int(S/2)
                    xDis = np.mean(patchFlowX[cS, cS])
                    yDis = np.mean(patchFlowY[cS, cS])
                    
                patchFlowX = patchFlowX - xDis
                patchFlowY = patchFlowY - yDis
                
                if np.sum(patchmask) == 0:

                    imgpath =  '%s%s%s%s%s%s%s' % (savePath, '/train/patch/', name, '_', str(j).zfill(2), str(i).zfill(2), '.png')
                    matpath =  '%s%s%s%s%s%s%s' % (savePath, '/train/patch_flow/', name, '_', str(j).zfill(2), str(i).zfill(2), '.npy')

                    io.imsave(imgpath, patch) 
                    np.save(matpath, np.array([patchFlowX, patchFlowY], dtype = np.float32))

generate(args.res, args.totalNum, args.savePath, args.dataPath)
moveTest(args.totalNum, args.testNum, args.savePath, args.dataPath)
