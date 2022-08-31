from datahandler import *
from autoaugment import CIFAR10Policy, Cutout
#import imagenet_policy as ip
#from thop import clever_format, profile

import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

# Added
import time
import sys
from scalable_senet import *
from torchvision import datasets
import os

def initialize(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
        elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


# Training
def train(epoch,trainloader):
    # print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        acc = 100. * correct / total
    return acc


def test(epoch,testloader):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            acc = 100. * correct / total

    return acc

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

torch.cuda.empty_cache()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Data transformations

initial_image_size = 32
total_classes = 10
number_input_channels = 3

print('==> Preparing data..')
transform_train = transforms.Compose(
    [
        transforms.RandomCrop(initial_image_size, padding=4),
        transforms.RandomHorizontalFlip(),
        CIFAR10Policy(),
        transforms.ToTensor(),
        Cutout(n_holes=1, length=16),  # (https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py)
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Architecture
print('==> Building network architecture..')
# d = []
# for i in range(4):
#     d.append(float(sys.argv[i+1]))

# w = []
# for i in range(4):
#     w.append(float(sys.argv[5+i]))
d=[2.0499999999999998224,1.9699999999999999734,1.8799999999999998934,3]
w=[0.78000000000000002665,1.2199999999999999734,0.80000000000000004441,0.67000000000000003997]
model = scaled_senet(d, w, initial_image_size)
model.to(device)
print(model)

if device == 'cuda':
    net = torch.nn.DataParallel(model)
    cudnn.benchmark = True

# Optimizer
print('==> Defining the Optimizer and its hyperparameters..')
# learning_rate=float(sys.argv[1])
# momentum=float(sys.argv[2])
# weight_decay=float(sys.argv[3])
# dampening=float(sys.argv[4])
# # learning_rate=0.042
# momentum=0.9
# weight_decay=0.005
# dampening=0
criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay,dampening=dampening)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=240, eta_min=1e-8)

# --------------------------------------------
# Dataset - Cifar10
# Plugin new dataset here
# --------------------------------------------

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)

y_train = trainset.targets
y_test = testset.targets

subset_indices_1, subset_indices_test_1 = get_subset_data(y_train, y_test)
partial_trainset = torch.utils.data.Subset(trainset, subset_indices_1)
partial_testset = torch.utils.data.Subset(testset, subset_indices_test_1)

# --------------------------------------------
# End of the dataset portion
# we need partial_trainset and partial_testset to define the trainloader and testloader
# --------------------------------------------

print('==> Model initialization..')
initialize(model)

trainloader = torch.utils.data.DataLoader(
    partial_trainset, batch_size=512, num_workers=2, shuffle=True)

testloader = torch.utils.data.DataLoader(
    partial_testset, batch_size=128, shuffle=False)

print(torch.version.cuda)
print(torch.cuda.is_available())
dict_accuracy={}


# start_epoch = 0
# training_accuracies = []
# testing_accuracies = []
# t0 = time.time()
# execution_time = 0
# total_epochs = 0
# epoch = 0
# best_test_acc = 0
# dataset="CIFAR10"
# print(dataset)
# while execution_time < 600:
#     tr_acc = train(epoch,trainloader)
#     training_accuracies.append(tr_acc)
#     te_acc = test(epoch,testloader)
#     testing_accuracies.append(te_acc)
#     if epoch <= 260:
#         scheduler.step()
#     if epoch==290 :
#         for param_group in optimizer.param_groups:
#             param_group['lr'] /= 10

#     execution_time = time.time() - t0

#     if te_acc > best_test_acc:
#         best_test_acc = te_acc
#         print('Saving checkpoint..')
#         state = {
#             'net': model.state_dict(),
#             'acc': best_test_acc,
#             'epoch': epoch,
#         }
#         torch.save(state, 'ckpt.pth')
#     lr = scheduler.get_last_lr()[0]

#     print(
#         "Epoch {}, Execution time: {:.1f}, LR: {:.3f}, Train accuracy: {:.3f}, Val accuracy: {:.3f} "
#             .format(epoch, execution_time, lr, tr_acc, best_test_acc))

#     epoch += 1

# print('Best valid acc CIFAR-10', max(testing_accuracies))
# print('Best train acc CIFAR-10', max(training_accuracies))
# dict_accuracy[dataset]=max(testing_accuracies)


data_dirs=[#"Mini-imagenet-0",
#"Mini-imagenet-1",
#"Mini-imagenet-2",
# "Mini-imagenet-3",
# "Mini-imagenet-4",
# "Mini-imagenet-5",
"Mini-imagenet-6",
# "Mini-imagenet-7",
# "Mini-imagenet-8",
"Mini-imagenet-9"]

for dataset in data_dirs:
   model = scaled_senet(d, w, initial_image_size)
   model.to(device)
   optimizer = optim.SGD(model.parameters(), lr=0.042, momentum=0.9, weight_decay=0.005)
   scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.5)
   initialize(model)
   train_loader, test_loader, _, val_data = prepare_imagenet(dataset)
   idx_to_class = {i: c for c, i in val_data.class_to_idx.items()}
   liste_best_accuracy=[]
   start_epoch = 0
   training_accuracies = []
   testing_accuracies = []
   t0 = time.time()
   execution_time = 0
   total_epochs = 0
   epoch = 0
   best_test_acc = 0
   print(dataset)
   while execution_time < 600:
       tr_acc = train(epoch,train_loader)
       training_accuracies.append(tr_acc)
       te_acc = test(epoch,test_loader)
       testing_accuracies.append(te_acc)
       if epoch >= 50:
        scheduler.step(te_acc)
  
    #    if epoch==50 :
    #        for param_group in optimizer.param_groups:
    #            param_group['lr'] /= 3
    #    if epoch==70:
    #         for param_group in optimizer.param_groups:
    #                     param_group['lr'] /= 3
    #    if epoch==90:
    #         for param_group in optimizer.param_groups:
    #                     param_group['lr'] /= 3

       execution_time = time.time() - t0

       if te_acc > best_test_acc:
           best_test_acc = te_acc
           print('Saving checkpoint..')
           state = {
               'net': model.state_dict(),
               'acc': best_test_acc,
               'epoch': epoch,
           }
           torch.save(state, 'ckpt.pth')
       #lr = scheduler.get_last_lr()[0]
       lr=0 

       print(
           "Epoch {}, Execution time: {:.1f}, LR: {:.3f}, Train accuracy: {:.3f}, Val accuracy: {:.3f} "
               .format(epoch, execution_time, lr, tr_acc, best_test_acc))

       epoch += 1

   print('Best valid acc '+dataset+' ', max(testing_accuracies))
   print('Best train acc '+dataset+' ', max(training_accuracies))
   dict_accuracy[dataset]=max(testing_accuracies)

print("------------------- Accuracy Nomad's Network --------------------------")
print(dict_accuracy)
res = 0
for val in dict_accuracy.values():
    res += val/len(dict_accuracy)
cnt=1
print("Moyenne_accuracy ",res)