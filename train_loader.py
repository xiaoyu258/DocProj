import os
from torch.utils import data
from torchvision import transforms
import scipy.io as spio
import numpy as np
import skimage
import torch

"""Custom Dataset compatible with prebuilt DataLoader."""
class DistortionDataset(data.Dataset):
    def __init__(self, patch_dir, flow_dir, global_dir, transform):
        
        self.local_img_paths = []
        self.flow_paths = []
        self.global_img_paths = []
            
        for fs in os.listdir(patch_dir):
            self.local_img_paths.append(os.path.join(patch_dir, fs))
        
        for fs in os.listdir(flow_dir):
            self.flow_paths.append(os.path.join(flow_dir, fs)) 

        for fs in os.listdir(global_dir):
            self.global_img_paths.append(os.path.join(global_dir, fs)) 
            
        self.local_img_paths.sort()
        self.flow_paths.sort()
        self.global_img_paths.sort()
        
        self.transform = transform
        
    def __getitem__(self, index):
        """Reads an image from a file and preprocesses it and returns."""
        local_img_path = self.local_img_paths[index]
        flow_path = self.flow_paths[index]
        global_img_path = self.global_img_paths[index]

        loal_img =skimage.io.imread(local_img_path)
        global_img = skimage.io.imread(global_img_path)
        flow = np.load(flow_path)
        
        flow = flow.astype(np.float32)

        if self.transform is not None:
            loal_img = self.transform(loal_img)
            global_img = self.transform(global_img)
   
        return loal_img, flow, global_img

    def __len__(self):
        """Returns the total number of image files."""
        return len(self.local_img_paths)
    
    
def get_loader(patch_dir, flow_dir, global_dir, batch_size):
    """Builds and returns Dataloader."""
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = DistortionDataset(patch_dir, flow_dir, global_dir, transform)
    data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    return data_loader