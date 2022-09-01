import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data
import random
import numpy as np
from autoaugment import *
import os
from torchvision import datasets

def get_class_i_indices(y, i):
    y = np.array(y)
    pos_i = np.argwhere(y == i)
    pos_i = list(pos_i[:, 0])
    random.shuffle(pos_i)

    return pos_i


def get_indices(dataset, class_name):
    indices = []
    for i in range(len(dataset.targets)):
        if dataset.targets[i] == class_name:
            indices.append(i)
    random.shuffle(indices)
    return indices


def dict_indices(dataset):
    idx_classes = {}
    for i in range(10):
        idx_classes[i] = get_indices(dataset, i)
    return idx_classes


def get_indx_balanced_train_subset(dict_indices, k):
    # print(len(dict_indices[0]))
    indx_balanced_subset = []
    for i in range(10):
        p10_idx = len(dict_indices[i]) // 10
        # print(p10_idx)
        indx_balanced_subset += dict_indices[i][k:k + p10_idx]
    return indx_balanced_subset


def get_indx_balanced_test_subset(dict_indices, k):
    indx_balanced_subset = []
    for i in range(10):
        indx_balanced_subset += dict_indices[i][k:k + 100]
    return indx_balanced_subset


def get_subset_data(y_train, y_test):

    classDict = {'plane': 0, 'car': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8,
                 'truck': 9}

    plane_indices = get_class_i_indices(y_train, classDict['plane'])
    car_indices = get_class_i_indices(y_train, classDict['car'])
    bird_indices = get_class_i_indices(y_train, classDict['bird'])
    cat_indices = get_class_i_indices(y_train, classDict['cat'])
    deer_indices = get_class_i_indices(y_train, classDict['deer'])
    dog_indices = get_class_i_indices(y_train, classDict['dog'])
    frog_indices = get_class_i_indices(y_train, classDict['frog'])
    horse_indices = get_class_i_indices(y_train, classDict['horse'])
    ship_indices = get_class_i_indices(y_train, classDict['ship'])
    truck_indices = get_class_i_indices(y_train, classDict['truck'])

    plane_indices_test = get_class_i_indices(y_test, classDict['plane'])
    car_indices_test = get_class_i_indices(y_test, classDict['car'])
    bird_indices_test = get_class_i_indices(y_test, classDict['bird'])
    cat_indices_test = get_class_i_indices(y_test, classDict['cat'])
    deer_indices_test = get_class_i_indices(y_test, classDict['deer'])
    dog_indices_test = get_class_i_indices(y_test, classDict['dog'])
    frog_indices_test = get_class_i_indices(y_test, classDict['frog'])
    horse_indices_test = get_class_i_indices(y_test, classDict['horse'])
    ship_indices_test = get_class_i_indices(y_test, classDict['ship'])
    truck_indices_test = get_class_i_indices(y_test, classDict['truck'])

    subset_indices_1 = plane_indices[0:500] + car_indices[0:500] + bird_indices[0:500] + cat_indices[0:500] + deer_indices[
                                                                                                          0:500] + dog_indices[
                                                                                                                   0:500] + frog_indices[
                                                                                                                            0:500] + horse_indices[
                                                                                                                                     0:500] + ship_indices[
                                                                                                                                              0:500] + truck_indices[
                                                                                                                                                       0:500]

    subset_indices_test_1 = plane_indices_test[0:100] + car_indices_test[0:100] + bird_indices_test[
                                                                              0:100] + cat_indices_test[
                                                                                       0:100] + deer_indices_test[
                                                                                                0:100] + dog_indices_test[
                                                                                                         0:100] + frog_indices_test[
                                                                                                                  0:100] + horse_indices_test[
                                                                                                                           0:100] + ship_indices_test[
                                                                                                                                    0:100] + truck_indices_test[
                                                                                                                                             0:100]
    return subset_indices_1, subset_indices_test_1
    trainset_1 = torch.utils.data.Subset(trainset, subset_indices_1)
    testset_1 = torch.utils.data.Subset(testset, subset_indices_test_1)

def prepare_imagenet(dataset):
    
    # Chemins 
    dataset_dirs={}
    dataset_dirs["Mini-imagenet-0"]="/home/dageloui/Documents/stage/challenge/challenge_septembre/Tiny-ImageNet/tiny-imagenet-0"
    dataset_dirs["Mini-imagenet-1"]="/home/dageloui/Documents/stage/challenge/challenge_septembre/Tiny-ImageNet/tiny-imagenet-1"
    dataset_dirs["Mini-imagenet-2"]="/home/dageloui/Documents/stage/challenge/challenge_septembre/Tiny-ImageNet/tiny-imagenet-2"
    dataset_dirs["Mini-imagenet-3"]="/home/dageloui/Documents/stage/challenge/challenge_septembre/Tiny-ImageNet/tiny-imagenet-3"
    dataset_dirs["Mini-imagenet-4"]="/home/dageloui/Documents/stage/challenge/challenge_septembre/Tiny-ImageNet/tiny-imagenet-4"
    dataset_dirs["Mini-imagenet-5"]="/home/dageloui/Documents/stage/challenge/challenge_septembre/Tiny-ImageNet/tiny-imagenet-5"
    dataset_dirs["Mini-imagenet-6"]="/home/dageloui/Documents/stage/challenge/challenge_septembre/Tiny-ImageNet/tiny-imagenet-6"
    dataset_dirs["Mini-imagenet-7"]="/home/dageloui/Documents/stage/challenge/challenge_septembre/Tiny-ImageNet/tiny-imagenet-7"
    dataset_dirs["Mini-imagenet-8"]="/home/dageloui/Documents/stage/challenge/challenge_septembre/Tiny-ImageNet/tiny-imagenet-8"
    dataset_dirs["Mini-imagenet-9"]="/home/dageloui/Documents/stage/challenge/challenge_septembre/Tiny-ImageNet/tiny-imagenet-9"
    
    
    train_dir = os.path.join(dataset_dirs[dataset], 'train')
    val_dir = os.path.join(dataset_dirs[dataset], 'val', 'images')
    initial_image_size = 32
    batch_size=256
    test_batch_size=128

    mean_train_norm_dir={}
    mean_train_norm_dir["CIFAR-10"]=[0.4914, 0.4822, 0.4465]
    mean_train_norm_dir["Mini-imagenet-0"]=[0.4755,0.4342,0.3585]
    mean_train_norm_dir["Mini-imagenet-1"]=[0.4601,0.4330,0.3734]
    mean_train_norm_dir["Mini-imagenet-2"]=[0.4626,0.4455,0.3916]
    mean_train_norm_dir["Mini-imagenet-3"]=[0.4682,0.4486,0.3834]
    mean_train_norm_dir["Mini-imagenet-4"]=[0.4698,0.4507,0.3801]
    mean_train_norm_dir["Mini-imagenet-5"]=[0.4713,0.4342,0.3585]
    mean_train_norm_dir["Mini-imagenet-6"]=[0.4725,0.4489,0.3834]
    mean_train_norm_dir["Mini-imagenet-7"]=[0.4744,0.4507,0.3903]
    mean_train_norm_dir["Mini-imagenet-8"]=[0.4736,0.4473,0.3900]
    mean_train_norm_dir["Mini-imagenet-9"]=[0.4742,0.4458,0.3910]
    
    std_train_norm_dir={}
    std_train_norm_dir["CIFAR-10"]=[0.2023, 0.1994, 0.2010]
    std_train_norm_dir["Mini-imagenet-0"]=[0.1889,0.1799,0.1722]
    std_train_norm_dir["Mini-imagenet-1"]=[0.1911,0.1849,0.1800]
    std_train_norm_dir["Mini-imagenet-2"]=[0.1955,0.1900,0.1863]
    std_train_norm_dir["Mini-imagenet-3"]=[0.1973,0.1917,0.1873]
    std_train_norm_dir["Mini-imagenet-4"]=[0.1976,0.1920,0.1872]
    std_train_norm_dir["Mini-imagenet-5"]=[0.1992,0.1940,0.1890]
    std_train_norm_dir["Mini-imagenet-6"]=[0.2022,0.1972,0.1931]
    std_train_norm_dir["Mini-imagenet-7"]=[0.2042,0.1993,0.1962]
    std_train_norm_dir["Mini-imagenet-8"]=[0.2061,0.2013,0.1984]
    std_train_norm_dir["Mini-imagenet-9"]=[0.2072,0.2027,0.2003]
    
    mean_val_norm_dir={}
    mean_val_norm_dir["CIFAR-10"]=[0.4914, 0.4822, 0.4465]
    mean_val_norm_dir["Mini-imagenet-0"]=[0.4782,0.4443,0.3706]
    mean_val_norm_dir["Mini-imagenet-1"]=[0.4589,0.4370,0.3805]
    mean_val_norm_dir["Mini-imagenet-2"]=[0.4587,0.4476,0.3947]
    mean_val_norm_dir["Mini-imagenet-3"]=[0.4662,0.4504,0.3871]
    mean_val_norm_dir["Mini-imagenet-4"]=[0.4683,0.4521,0.3811]
    mean_val_norm_dir["Mini-imagenet-5"]=[0.4709,0.4529,0.3824]
    mean_val_norm_dir["Mini-imagenet-6"]=[0.4718,0.4502,0.3845]
    mean_val_norm_dir["Mini-imagenet-7"]=[0.4727,0.4509,0.3900]
    mean_val_norm_dir["Mini-imagenet-8"]=[0.4720,0.4472,0.3893]
    mean_val_norm_dir["Mini-imagenet-9"]=[0.4734,0.4462,0.3908]
    
    std_val_norm_dir={}
    std_val_norm_dir["CIFAR-10"]=[0.2023, 0.1994, 0.2010]
    std_val_norm_dir["Mini-imagenet-0"]=[0.1885,0.1801,0.1733]
    std_val_norm_dir["Mini-imagenet-1"]=[0.1890,0.1829,0.1787]
    std_val_norm_dir["Mini-imagenet-2"]=[0.1933,0.1883,0.1856]
    std_val_norm_dir["Mini-imagenet-3"]=[0.1961,0.1904,0.1868]
    std_val_norm_dir["Mini-imagenet-4"]=[0.1968,0.1908,0.1862]
    std_val_norm_dir["Mini-imagenet-5"]=[0.1980,0.1924,0.1874]
    std_val_norm_dir["Mini-imagenet-6"]=[0.2012,0.1959,0.1916]
    std_val_norm_dir["Mini-imagenet-7"]=[0.2033,0.1983,0.1951]
    std_val_norm_dir["Mini-imagenet-8"]=[0.2056,0.2003,0.1973]
    std_val_norm_dir["Mini-imagenet-9"]=[0.2067,0.2019,0.1995]
    
    mean_train=mean_train_norm_dir[dataset]
    std_train=std_train_norm_dir[dataset]
    mean_val=mean_val_norm_dir[dataset]
    std_val=std_val_norm_dir[dataset]

    #imagenet_policy=ip.AutoAugment()

    train_transform = transforms.Compose(
    [
        transforms.RandomCrop(initial_image_size, padding=4),
        transforms.RandomHorizontalFlip(),
        CIFAR10Policy(),
        transforms.ToTensor(),
        Cutout(n_holes=1, length=16),  # (https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py)
        transforms.Normalize(mean=mean_train, std=std_train),
        ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_val, std=std_val),
    ])


    train_data = datasets.ImageFolder(train_dir, 
                                    transform=train_transform)
    
    val_data = datasets.ImageFolder(val_dir, 
                                    transform=val_transform)
    
    #print('Preparing data loaders ...')
    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, 
                                                    shuffle=True, num_workers=1,pin_memory=True)
    
    val_data_loader = torch.utils.data.DataLoader(val_data, batch_size=test_batch_size, 
                                                    shuffle=True, num_workers=1)
    
    return train_data_loader, val_data_loader, train_data, val_data
