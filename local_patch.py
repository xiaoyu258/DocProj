import numpy as np
from skimage import transform as tf
from skimage import io
import os
import skimage
import shutil

import warnings
warnings.filterwarnings('ignore')

def createDir(savePath):
    
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    
    path =  '%s%s' % (savePath, '/train/img')
    if not os.path.exists(path):
        os.makedirs(path)
        
    path =  '%s%s' % (savePath, '/train/flow')
    if not os.path.exists(path):
        os.makedirs(path)

    path =  '%s%s' % (savePath, '/test/img')
    if not os.path.exists(path):
        os.makedirs(path)
        
    path =  '%s%s' % (savePath, '/test/flow')
    if not os.path.exists(path):
        os.makedirs(path)
        
def moveTest(num, testNum, savePath, datasetpath):
    
    testDir = os.listdir('%s%s' % (datasetpath, '/0_img_bg'))[(num - testNum):num]
    testDir.sort()
    
    imgPaths = os.listdir('%s%s' % (savePath, '/train/img'))
    imgPaths.sort()
    
    for fs in imgPaths:
        
        testIndex = '%s%s' % (fs[0:5], '.png')
        
        if testIndex in testDir:

            name = fs.split('.')[0]

            dir1 = '%s%s%s%s' % (savePath, '/train/img/', name, '.png')
            dir2 = '%s%s%s%s' % (savePath, '/test/img/', name, '.png')
            shutil.move(dir1, dir2)

            dir5 = '%s%s%s%s' % (savePath, '/train/flow/', name, '.npy')
            dir6 = '%s%s%s%s' % (savePath, '/test/flow/', name, '.npy')
            shutil.move(dir5, dir6)

def generate(S, num, savePath, datasetpath):
    
    createDir(savePath)
    
    ind = 0

    for fs in os.listdir('%s%s' % (datasetpath, '/0_img_bg'))[0:num]:

        name = fs.split('.')[0]
        
        print(ind, name)
        ind += 1

        imgpath =  '%s%s%s%s' % (datasetpath, '/0_img_bg/', name, '.png')
        mskpath =  '%s%s%s%s' % (datasetpath, '/0_img_msk/', name, '.png')
        matpath =  '%s%s%s%s' % (datasetpath, '/0_flow_npy/', name, '.npy')

        image = io.imread(imgpath)
        mask = io.imread(mskpath)
        flow  = np.load(matpath)

        H = image.shape[0]
        W = image.shape[1]
        
        cropH = S
        cropW = S
        ovlp = int(S * 0)
        
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
                    
                if xDis == 0 and yDis == 0:
                    xDis = np.mean(patchFlowX[patchmask[:,:] == 0])
                    yDis = np.mean(patchFlowY[patchmask[:,:] == 0])
                    
                patchFlowX = patchFlowX - xDis
                patchFlowY = patchFlowY - yDis
                
                patchFlowX[patchmask[:,:] != 0] = 0
                patchFlowY[patchmask[:,:] != 0] = 0
                
                if np.sum(patchmask) == 0:

                    imgpath =  '%s%s%s%s%s%s%s' % (savePath, '/train/img/', name, '_', str(j).zfill(2), str(i).zfill(2), '.png')
                    matpath =  '%s%s%s%s%s%s%s' % (savePath, '/train/flow/', name, '_', str(j).zfill(2), str(i).zfill(2), '.npy')

                    io.imsave(imgpath, patch) 
                    np.save(matpath, np.array([patchFlowX, patchFlowY], dtype = np.float32))

S = 256         # the resolution of local patch
num = 2750      # the number of images to be cropped for the dataset
testNum = 250   # the number of images to be cropped for validation dataset
datasetpath = '/home/xliea/DocProj/dataset'                      # the image path
savePath    = '%s%s' % ('/home/xliea/DocProj/dataset_patch', S)  # the patch dataset path to be saved
generate(S, num, savePath, datasetpath)
moveTest(num, testNum, savePath, datasetpath)  
