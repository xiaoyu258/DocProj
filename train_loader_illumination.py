import os
from torch.utils import data
from torchvision import transforms
import scipy.io as spio
import numpy as np
import skimage
import torch
from PIL import Image

"""Custom Dataset compatible with prebuilt DataLoader."""
class DistortionDataset(data.Dataset):
    def __init__(self, distorted_image_dir, corrected_image_dir, transform):
        
        self.distorted_image_paths = []
        self.corrected_image_paths = []
            
        for fs in os.listdir(distorted_image_dir):
            self.distorted_image_paths.append(os.path.join(distorted_image_dir, fs)) 
        
        for fs in os.listdir(corrected_image_dir):
            self.corrected_image_paths.append(os.path.join(corrected_image_dir, fs)) 

        self.distorted_image_paths.sort()
        self.corrected_image_paths.sort()
        self.transform = transform
        
    def __getitem__(self, index):
        """Reads an image from a file and preprocesses it and returns."""
        distorted_image_path = self.distorted_image_paths[index]
        corrected_image_path = self.corrected_image_paths[index]
        
        distorted_image = skimage.io.imread(distorted_image_path)
        distorted_image = distorted_image.astype(np.float32)/255.0
        distorted_image = torch.Tensor(distorted_image).permute(2, 0, 1)
        
        corrected_image = skimage.io.imread(corrected_image_path)
        corrected_image = corrected_image.astype(np.float32)/255.0
        corrected_image = torch.Tensor(corrected_image).permute(2, 0, 1)
        
        tfImg = self.transform(distorted_image)
   
        return tfImg, corrected_image

    def __len__(self):
        """Returns the total number of image files."""
        return len(self.distorted_image_paths)
    
    
def get_loader(distorted_image_dir, corrected_image_dir, batch_size):
    """Builds and returns Dataloader."""
    
    transform = transforms.Compose([transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    dataset = DistortionDataset(distorted_image_dir, corrected_image_dir, transform)
    data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    return data_loader