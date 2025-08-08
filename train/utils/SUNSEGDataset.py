from PIL import Image
import os
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
import torch
from sam2.build_sam import build_sam2
from typing import List, Optional, Tuple
from sam2.modeling.sam2_base import SAM2Base
from sam2.utils.transforms import SAM2Transforms

class SUNSEGDataset(Dataset):
    def __init__(self, data_root, memory_length=8):
        self.data_root = data_root
        self.case_path = os.path.join(self.data_root, 'Frame') 
        self.case_list = [d for d in os.listdir(self.case_path) if os.path.isdir(os.path.join(self.case_path, d))]
        print(self.case_list)
        self.memory_length = memory_length
        self.rng = np.random.default_rng(seed=42)
        self.data_by_cases = {}
        # self.step_size = 5
        self.step_size = 1

        for case in self.case_list:
            print(case)
            img_dir = os.path.join(self.data_root, 'Frame', case)
            gt_dir = os.path.join(self.data_root, 'GT', case)

            tmp_list = os.listdir(img_dir)
            tmp_list.sort(key=lambda name: (
                int(name.split('-')[0].split('_')[-1]),
                int(name.split('_a')[1].split('_')[0]),
                int(name.split('_image')[1].split('.jpg')[0])))
            
            
            gt_filenames = []
            img_filenames = []
            for filename in tmp_list:
                gt_filenames.append(os.path.join(gt_dir, filename.replace(".jpg", ".png")))
                img_filenames.append(os.path.join(img_dir, filename))
            #img_filenames = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.jpg')])
            #gt_filenames = sorted([os.path.join(gt_dir, f) for f in os.listdir(gt_dir) if f.endswith('.png')])

            # Store the images and corresponding ground truth by case
            self.data_by_cases[case] = {
                'imgs': img_filenames,
                'gts': gt_filenames
            }

        self._transform = SAM2Transforms(resolution=1024, mask_threshold=0)
        self.new_size = (1024, 1024)
        
        # Create a list of valid indices for sampling
        self.valid_indices = []
        for case, data in self.data_by_cases.items():
            num_frames = len(data['imgs'])
            # ensure have enough length
            if num_frames > self.memory_length * self.step_size:
                self.valid_indices.extend([(case, i) for i in range(0, num_frames-(self.memory_length-1) * self.step_size)])


    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, index):
        case, frame_idx = self.valid_indices[index]
        
        img_files = self.data_by_cases[case]['imgs']
        gt_files = self.data_by_cases[case]['gts']
    
        # Load current frame's image and ground truth
        current_img = Image.open(img_files[frame_idx]).convert('RGB').resize(self.new_size)
        current_gt = Image.open(gt_files[frame_idx]).convert('L').resize((256, 256), Image.NEAREST)
    
        current_img = self._transform(np.array(current_img))

        current_gt = torch.from_numpy(np.array(current_gt).astype(float)).unsqueeze(0)  # Add channel dimension to current_gt

        
        memory_indices = []
        for i in range(self.memory_length):
            memory_indices.append(frame_idx + self.step_size * i)
    
        if self.rng.random() < 0.5:
            memory_indices = memory_indices[::-1]
            
        memory_imgs, memory_gts = [], []
        for i in memory_indices:
            mem_img = Image.open(img_files[i]).convert('RGB').resize(self.new_size)
            mem_gt = Image.open(gt_files[i]).convert('L').resize((256, 256), Image.NEAREST)
    
            memory_imgs.append(self._transform(np.array(mem_img)))
            mem_gt = np.array(mem_gt) > 128
            memory_gts.append(torch.from_numpy(mem_gt.astype(float)).unsqueeze(0))
    
        # Stack memory images and labels
        memory_imgs = torch.stack(memory_imgs)  # Shape: (memory_length, 3, H, W)
        memory_gts = torch.stack(memory_gts)  # Shape: (memory_length, 1, H, W)
    
        return {
            "current_image": current_img,
            "current_label": current_gt,  # Shape: (1, H, W)
            "memory_images": memory_imgs,  # Shape: (memory_length, 3, H, W)
            "memory_labels": memory_gts  # Shape: (memory_length, 1, H, W)
        }
