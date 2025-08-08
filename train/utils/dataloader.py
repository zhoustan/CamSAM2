import numpy as np
import random
from copy import deepcopy
from skimage import io
import os
from glob import glob

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms, utils
from torchvision.transforms.functional import normalize
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler

class LargeScaleJitter(object):
    """
        implementation of large scale jitter from copy_paste
        https://github.com/gaopengcuhk/Pretrained-Pix2Seq/blob/7d908d499212bfabd33aeaa838778a6bfb7b84cc/datasets/transforms.py 
    """

    def __init__(self, output_size=1024, aug_scale_min=0.1, aug_scale_max=2.0):
        self.desired_size = torch.tensor(output_size)
        self.aug_scale_min = aug_scale_min
        self.aug_scale_max = aug_scale_max

    def __call__(self, sample):
        images, labels = sample['memory_images'], sample['memory_labels']
        image_size = torch.tensor(images[0].shape[-2:])
        assert images.ndim == 4 
        assert labels.ndim == 4
        # Resize while keeping aspect ratio
        out_desired_size = (self.desired_size * image_size / max(image_size)).round().int()
        random_scale = torch.rand(1) * (self.aug_scale_max - self.aug_scale_min) + self.aug_scale_min
        scaled_size = (random_scale * self.desired_size).round()
        
        scale = torch.minimum(scaled_size / image_size[0], scaled_size / image_size[1])
        scaled_size = (image_size * scale).round().long()

        # Resize all images and labels in a single operation
        images = F.interpolate(images, size=scaled_size.tolist(), mode='bilinear', align_corners=False)
        labels = F.interpolate(labels, size=scaled_size.tolist(), mode='nearest')  # Use 'nearest' for labels

        # Random crop
        crop_size = (min(self.desired_size, scaled_size[0]), min(self.desired_size, scaled_size[1]))
        margin_h = max(scaled_size[0] - crop_size[0], 0).item()
        margin_w = max(scaled_size[1] - crop_size[1], 0).item()
        
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        crop_y1, crop_y2 = offset_h, offset_h + crop_size[0].item()
        crop_x1, crop_x2 = offset_w, offset_w + crop_size[1].item()

        # Apply cropping in a single operation
        images = images[:, :, crop_y1:crop_y2, crop_x1:crop_x2]
        labels = labels[:, :, crop_y1:crop_y2, crop_x1:crop_x2]

        # Padding to desired size if needed
        padding_h = max(self.desired_size - images.shape[2], 0)
        padding_w = max(self.desired_size - images.shape[3], 0)

        images = F.pad(images, [0, padding_w, 0, padding_h], value=128)
        labels = F.pad(labels, [0, padding_w, 0, padding_h], value=0)

        return {
            "memory_images": images,
            "memory_labels": labels,
            "current_image": sample['current_image'],
            "current_label": sample['current_label'],
        }

class RandomHFlip(object):
    def __init__(self,prob=0.5):
        self.prob = prob
    def __call__(self,sample):
        images, labels =  sample['memory_images'], sample['memory_labels']

        # random horizontal flip
        if random.random() >= self.prob:
            print(True)
            # Flip along width dimension (dim=3)
            images = torch.flip(images, dims=[3])
            labels = torch.flip(labels, dims=[3])
        else:
            print(False)

        return {
            "memory_images": images,
            "memory_labels": labels,
            "current_image": sample['current_image'],
            "current_label": sample['current_label'],
        }