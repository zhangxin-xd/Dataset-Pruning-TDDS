import time
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms

########################################################################################################################
# Load Data
########################################################################################################################

def load_data(args):
    """
    Load data for training and testing.

    Returns:
        train_loader: DataLoader for training data.
        test_loader: DataLoader for testing data.
    """
    train_loader, test_loader = load_dataset(args)
    return train_loader, test_loader

def load_dataset(args):
    """
    Load dataset based on the specified dataset in args.

    Returns:
        train_loader: DataLoader for training data.
        test_loader: DataLoader for testing data.
    """
    if args.dataset == 'cifar10':
        train_loader, test_loader = load_cifar10(args)
    elif args.dataset == 'cifar100':
        train_loader, test_loader = load_cifar100(args)
    else:
        raise NotImplementedError("Dataset not supported: {}".format(args.dataset))
    return train_loader, test_loader

def load_cifar10(args):
    """
    Load CIFAR-10 dataset.

    Returns:
        train_loader: DataLoader for training data.
        test_loader: DataLoader for testing data.
    """
    print('Loading CIFAR-10... ', end='')
    time_start = time.time()
    
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
    
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    train_data = dset.CIFAR10(args.data_path, train=True, transform=train_transform, download=True)
    target_index = [[train_data.targets[i], i] for i in range(len(train_data.targets))]
    train_data.targets = target_index
    
    train_loader = torch.utils.data.DataLoader(train_data, args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True)
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    test_data = dset.CIFAR10(args.data_path, train=False, transform=test_transform, download=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.workers, pin_memory=True)
    
    print(f"done in {time.time() - time_start:.2f} seconds.")
    return train_loader, test_loader

def load_cifar100(args):
    """
    Load CIFAR-100 dataset.

    Returns:
        train_loader: DataLoader for training data.
        test_loader: DataLoader for testing data.
    """
    print('Loading CIFAR-100... ', end='')
    time_start = time.time()
    
    mean = [x / 255 for x in [129.3, 124.1, 112.4]]
    std = [x / 255 for x in [68.2, 65.4, 70.4]]
    
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    train_data = dset.CIFAR100(args.data_path, train=True, transform=train_transform, download=True)
    target_index = [[train_data.targets[i], i] for i in range(len(train_data.targets))]
    train_data.targets = target_index
    
    train_loader = torch.utils.data.DataLoader(train_data, args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True)
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    test_data = dset.CIFAR100(args.data_path, train=False, transform=test_transform, download=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.workers, pin_memory=True)
    
    print(f"done in {time.time() - time_start:.2f} seconds.")
    return train_loader, test_loader
