import torch
import torch.nn as nn
from torch.autograd import Variable
import skimage
from skimage import io
import numpy as np
import argparse
from torchvision import transforms
import os

from logger import Logger
from train_loader_illumination import get_loader
from model_illNet import illNet
from vgg import VGG19

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR')
parser.add_argument('--epochs', type=int, default=2, metavar='N')
parser.add_argument('--batch_size', type=int, default=32, metavar='N')
parser.add_argument('--dataset_dir', type=str, default='./dataset', help='dataset path')
parser.add_argument("--savemodel_dir", type=str, default='./model.pkl', help='save model path')
args = parser.parse_args()

train_loader = get_loader(distorted_image_dir = '%s%s' % (args.dataset_dir, '/train/blur'),
                  corrected_image_dir = '%s%s' % (args.dataset_dir, '/train/rect'), 
                  batch_size = args.batch_size)

model = illNet()
vggnet = VGG19()
vggnet.load_state_dict(torch.load('./vgg19.pkl'))
for param in vggnet.parameters():
    param.requires_grad = False
    
criterion = nn.L1Loss()
vgg_loss = nn.L1Loss()

if torch.cuda.is_available():
    model = model.cuda()
    vggnet = vggnet.cuda()
    criterion = criterion.cuda()
    vgg_loss = vgg_loss.cuda()
    
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

lr = args.lr
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

step = 0
logger = Logger('./logs')

model.train()
for epoch in range(args.epochs):
    for i, (blur, rect) in enumerate(train_loader):
         
        if torch.cuda.is_available():
            blur = blur.cuda()
            rect = rect.cuda()
        
        blur = Variable(blur)
        rect = Variable(rect)
        
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        output = model(blur)

        image_loss = criterion(output, rect)
        
        A_relu44 = vggnet(rect, ['p5'], preprocess=False)[0]
        B_relu44 = vggnet(output, ['p5'], preprocess=False)[0]
        perception_loss = vgg_loss(A_relu44, B_relu44)*1e-5
        
        loss = image_loss + perception_loss

        loss.backward()
        optimizer.step()
        print("Epoch [%d], Iter [%d], Loss: %.4f, Loss: %.4f, Loss: %.4f" % 
              (epoch + 1, i + 1, loss.item(), image_loss.item(), perception_loss.item()))
        #============ TensorBoard logging ============#
        step = step + 1
        
        #Log the scalar values
        info = {'loss': loss.item()}
        for tag, value in info.items():
            logger.scalar_summary(tag, value, step)
        
    torch.save(model.state_dict(), args.savemodel_dir)
