""" configurations for this project

author baiyu
"""
import torch
from datetime import datetime

# CIFAR100 dataset path (python version)
# CIFAR100_PATH = '/nfs/private/cifar100/cifar-100-python'
CIFAR100_PATH = 'D:/Data/cifar-100-python/'
CIFAR10_PATH = 'D:/Data/cifar-10-python/'

# mean and std of cifar100 dataset
CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

# CIFAR100_TEST_MEAN = (0.5088964127604166, 0.48739301317401956, 0.44194221124387256)
# CIFAR100_TEST_STD = (0.2682515741720801, 0.2573637364478126, 0.2770957707973042)

# mean and std of cifar10 dataset
CIFAR10_TRAIN_MEAN = (0.49139968, 0.48215841, 0.44653091)
CIFAR10_TRAIN_STD = (0.24703223, 0.24348513, 0.26158784)

# CIFAR10_TEST_MEAN = (0.5088964127604166, 0.48739301317401956, 0.44194221124387256)
# CIFAR10_TEST_STD = (0.2682515741720801, 0.2573637364478126, 0.2770957707973042)

# mean and std of MINST dataset
MNIST_TRAIN_MEAN = (0.1307,)
MNIST_TRAIN_STD = (0.3081,)

# mean and std of Tiny_ImageNet
TINY_IMAGENET_MEAN = [0.4802, 0.4481, 0.3975]
TINY_IMAGENET_STD = [0.2302, 0.2265, 0.2262]

# mean and std of Caltech256
CALTECH_256_MEAN = [0.485, 0.456, 0.406]
CALTECH_256_STD = [0.229, 0.224, 0.225]

# directory to save weights file
CHECKPOINT_PATH = 'checkpoint/CIFAR100'

# total training epoches
EPOCH = 200
# MILESTONES = [100, 225]
MILESTONES = [60, 120, 160]
# MILESTONES = [50, 100, 150]

# initial learning rate
# INIT_LR = 0.1

# time of we run the script
TIME_NOW = datetime.now().strftime('%A_%d_%B_%Y_%Hh_%Mm_%Ss')

# tensorboard log dir
LOG_DIR = 'runs'

# save weights file per SAVE_EPOCH epoch
SAVE_EPOCH = 10

CUDA = 'cuda:0' if torch.cuda.is_available() else None
