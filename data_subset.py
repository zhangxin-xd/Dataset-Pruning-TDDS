import numpy as np
import time
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from multiprocessing import Pool
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

########################################################################################################################
#  Load Data
########################################################################################################################

def load_cifar10_sub(args, data_mask, sorted_score):
    """
    Load CIFAR10 dataset with specified transformations and subset selection.
    """
    print('Loading CIFAR10... ', end='')
    time_start = time.time()
    
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
    
    score = (sorted_score - min(sorted_score)) / (max(sorted_score) - min(sorted_score))
    
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    train_data = dset.CIFAR10(args.data_path, train=True, transform=train_transform, download=True)
    z = [[train_data.targets[i], score[np.where(data_mask == i)]] for i in range(len(train_data.targets))]
    train_data.targets = z

    subset_mask = data_mask[int(args.subset_rate * len(data_mask)):]
    data_set = torch.utils.data.Subset(train_data, subset_mask)

    train_loader = torch.utils.data.DataLoader(data_set, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True)

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    test_data = dset.CIFAR10(args.data_path, train=False, transform=test_transform, download=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.workers, pin_memory=True)
    return train_loader, test_loader

def load_cifar100_sub(args, data_mask, sorted_score):
    """
    Load CIFAR100 dataset with specified transformations and subset selection.
    """
    print('Loading CIFAR100... ', end='')
    time_start = time.time()
    
    mean = [x / 255 for x in [129.3, 124.1, 112.4]]
    std = [x / 255 for x in [68.2, 65.4, 70.4]]
    
    score = (sorted_score - min(sorted_score)) / (max(sorted_score) - min(sorted_score))
    
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    train_data = dset.CIFAR100(args.data_path, train=True, transform=train_transform, download=True)
    z = [[train_data.targets[i], score[np.where(data_mask == i)]] for i in range(len(train_data.targets))]
    train_data.targets = z

    subset_mask = data_mask[int(args.subset_rate * len(data_mask)):]
    data_set = torch.utils.data.Subset(train_data, subset_mask)

    train_loader = torch.utils.data.DataLoader(data_set, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True)

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    test_data = dset.CIFAR100(args.data_path, train=False, transform=test_transform, download=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.workers, pin_memory=True)
    return train_loader, test_loader
