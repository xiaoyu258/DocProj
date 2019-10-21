import numpy as np
import skimage.io as io
from numba import cuda
import math
import argparse

parser = argparse.ArgumentParser(description='resamping')
parser.add_argument("--img_path", type=str, default= '/home/xliea/GeoProj/img.png')
parser.add_argument("--flow_path", type=str, default= '/home/xliea/GeoProj/flow.npy')
args = parser.parse_args()

@cuda.jit(device=True)
def iterSearchShader(padu, padv, xr, yr, maxIter, precision):
    
    H = padu.shape[0] - 1
    W = padu.shape[1] - 1
    
    if abs(padu[yr,xr]) < precision and abs(padv[yr,xr]) < precision:
        return xr, yr

    else:
        # Our initialize method in this paper, can see the overleaf for detail
        if (xr + 1) <= (W - 1):
            dif = padu[yr,xr + 1] - padu[yr,xr]
            u_next = padu[yr,xr]/(1 + dif)
        else:
            dif = padu[yr,xr] - padu[yr,xr - 1]
            u_next = padu[yr,xr]/(1 + dif)

        if (yr + 1) <= (H - 1):
            dif = padv[yr + 1,xr] - padv[yr,xr]
            v_next = padv[yr,xr]/(1 + dif)
        else:
            dif = padv[yr,xr] - padv[yr - 1,xr]
            v_next = padv[yr,xr]/(1 + dif)

        i = xr - u_next
        j = yr - v_next
        '''
        i = xr - padu[yr,xr]
        j = yr - padv[yr,xr]
        '''
        # The same as traditinal iterative search method
        for iter in range(maxIter):

            if 0<= i <= (W - 1) and 0 <= j <= (H - 1):

                u11 = padu[int(j), int(i)]
                v11 = padv[int(j), int(i)]

                u12 = padu[int(j), int(i) + 1]
                v12 = padv[int(j), int(i) + 1]

                u21 = padu[int(j) + 1, int(i)]
                v21 = padv[int(j) + 1, int(i)]

                u22 = padu[int(j) + 1, int(i) + 1]
                v22 = padv[int(j) + 1, int(i) + 1]


                u = u11*(int(i) + 1 - i)*(int(j) + 1 - j) + u12*(i - int(i))*(int(j) + 1 - j) + \
                    u21*(int(i) + 1 - i)*(j - int(j)) + u22*(i - int(i))*(j - int(j))

                v = v11*(int(i) + 1 - i)*(int(j) + 1 - j) + v12*(i - int(i))*(int(j) + 1 - j) + \
                    v21*(int(i) + 1 - i)*(j - int(j)) + v22*(i - int(i))*(j - int(j))

                i_next = xr - u
                j_next = yr - v                

                if abs(i - i_next)<precision and abs(j - j_next)<precision:
                    return i, j

                i = i_next
                j = j_next

            else:     
                return -1, -1
        '''
        return -1, -1
        '''
        # if the search doesn't converge within max iter, it will return the last iter result
        if 0 <= i_next <= (W - 1) and 0 <= j_next <= (H - 1):
            return i_next, j_next
        elif 0 <= i <= (W - 1) and 0 <= j <= (H - 1):
            return i, j
        else:
            return -1, -1
        

            
@cuda.jit(device=True)
def biInterpolation(distorted, i, j):
    Q11 = distorted[int(j), int(i)]
    Q12 = distorted[int(j), int(i) + 1]
    Q21 = distorted[int(j) + 1, int(i)]
    Q22 = distorted[int(j) + 1, int(i) + 1]
    pixel = Q11*(int(i) + 1 - i)*(int(j) + 1 - j) + Q12*(i - int(i))*(int(j) + 1 - j) + \
            Q21*(int(i) + 1 - i)*(j - int(j)) + Q22*(i - int(i))*(j - int(j))
    return pixel


@cuda.jit
def iterSearch(padu, padv, paddistorted, resultImg, maxIter, precision, resultMsk):
    
    H = padu.shape[0] - 1
    W = padu.shape[1] - 1
    
    start_x, start_y = cuda.grid(2)
    stride_x, stride_y = cuda.gridsize(2)
    
    for xr in range(start_x, W, stride_x):
        for yr in range(start_y, H, stride_y):

            i,j = iterSearchShader(padu, padv, xr, yr, maxIter, precision)

            if(i != -1) and (j != -1):
                resultImg[yr, xr,0] = biInterpolation(paddistorted[:,:,0], i, j)
                resultImg[yr, xr,1] = biInterpolation(paddistorted[:,:,1], i, j)
                resultImg[yr, xr,2] = biInterpolation(paddistorted[:,:,2], i, j)
            else:
                resultMsk[yr, xr] = 255

@cuda.jit
def iterSearchGrey(padu, padv, paddistorted, resultImg, maxIter, precision, resultMsk):
    
    H = padu.shape[0] - 1
    W = padu.shape[1] - 1
  
    start_x, start_y = cuda.grid(2)
    stride_x, stride_y = cuda.gridsize(2)
    
    for xr in range(start_x, W, stride_x):
        for yr in range(start_y, H, stride_y):

            i,j = iterSearchShader(padu, padv, xr, yr, maxIter, precision)

            if(i != -1) and (j != -1):
                resultImg[yr, xr] = biInterpolation(paddistorted[:,:], i, j)
            else:
                resultMsk[yr, xr] = 255

def rectification(distorted, flow):
    
    H = distorted.shape[0]
    W = distorted.shape[1]

    maxIter = 100
    precision = 1e-2

    isGrey = True
    resultMsk = np.array(np.zeros((H, W)), dtype = np.uint8)
    if len(distorted.shape) == 3:
        resultImg = np.array(np.zeros((H, W, 3)), dtype = np.uint8)
        paddistorted = np.array(np.zeros((H + 1, W + 1, 3)), dtype = np.uint8)
        resultImg.fill(255)
        isGrey = False
    else:
        resultImg = np.array(np.zeros((H, W)), dtype = np.uint8)
        paddistorted = np.array(np.zeros((H + 1, W + 1)), dtype = np.uint8)
        resultImg.fill(255)
        isGrey = True

    paddistorted[0:H, 0:W] = distorted[0:H, 0:W]
    paddistorted[H, 0:W] = distorted[H-1, 0:W]
    paddistorted[0:H, W] = distorted[0:H, W-1]
    paddistorted[H, W] = distorted[H-1, W-1]

    padu = np.array(np.zeros((H + 1, W + 1)), dtype = np.float32)
    padu[0:H, 0:W] = flow[0][0:H, 0:W]
    padu[H, 0:W] = flow[0][H-1, 0:W]
    padu[0:H, W] = flow[0][0:H, W-1]
    padu[H, W] = flow[0][H-1, W-1]

    padv = np.array(np.zeros((H + 1, W + 1)), dtype = np.float32)
    padv[0:H, 0:W] = flow[1][0:H, 0:W]
    padv[H, 0:W] = flow[1][H-1, 0:W]
    padv[0:H, W] = flow[1][0:H, W-1]
    padv[H, W] = flow[1][H-1, W-1]

    padu = cuda.to_device(padu)
    padv = cuda.to_device(padv)
    paddistorted = cuda.to_device(paddistorted)
    resultImg = cuda.to_device(resultImg)
    resultMsk = cuda.to_device(resultMsk)

    threadsperblock = (16, 16)
    blockspergrid_x = math.ceil(W / threadsperblock[0])
    blockspergrid_y = math.ceil(H / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    if isGrey:
        iterSearchGrey[blockspergrid, threadsperblock](padu, padv, paddistorted, resultImg, maxIter, precision, resultMsk)
    else:
        iterSearch[blockspergrid, threadsperblock](padu, padv, paddistorted, resultImg, maxIter, precision, resultMsk)

    resultImg = resultImg.copy_to_host()
    resultMsk = resultMsk.copy_to_host()
    
    return resultImg, resultMsk
    
distortedImg = io.imread(args.img_path)  
flow = np.load(args.flow_path)
resImg, resMsk = rectification(distortedImg, flow)
io.imsave(resImg, 'result.png')
