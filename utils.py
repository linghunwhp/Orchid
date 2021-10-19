""" helper function

author baiyu
"""

import sys
import random
import numpy
import torch
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, SubsetRandomSampler, random_split, Dataset, Subset
from conf import settings
from dataset import TinyImagenet200
import numpy as np


def get_network(args):
    """ return given network
    """
    if args.net == 'allcnn':
        from models.allcnn import allcnn
        net = allcnn(num_class=args.num_class)
    elif args.net == 'cnn':
        from models.cnn import cnn
        net = cnn(num_class=args.num_class)
    elif args.net == 'classify_nn':
        from models.classify_nn import classify_nn
        net = classify_nn(num_inputs=args.num_inputs, num_hiddens=args.hiddens, num_class=args.num_class)
    elif args.net == 'classify_softmax':
        from models.classify_softmax import classify_softmax
        net = classify_softmax(num_inputs=args.num_inputs, num_class=args.num_class)
    elif args.net == 'vgg16':
        from models.vgg import vgg16_bn
        net = vgg16_bn(num_class=args.num_class)
    elif args.net == 'vgg13':
        from models.vgg import vgg13_bn
        net = vgg13_bn(num_class=args.num_class)
    elif args.net == 'vgg11':
        from models.vgg import vgg11_bn
        net = vgg11_bn(num_class=args.num_class)
    elif args.net == 'vgg19':
        from models.vgg import vgg19_bn
        net = vgg19_bn(num_class=args.num_class)
    elif args.net == 'densenetbc':
        from models.densenetbc import densenetbc
        net = densenetbc(growthRate=12, depth=100, reduction=0.5, bottleneck=True, num_class=args.num_class)
        # net = densenetbc(growthRate=24, depth=250, reduction=0.5, bottleneck=True, num_class=args.num_class)
    elif args.net == 'densenet121':
        from models.densenet import densenet121
        net = densenet121(num_class=args.num_class)
    elif args.net == 'densenet161':
        from models.densenet import densenet161
        net = densenet161(num_class=args.num_class)
    elif args.net == 'densenet169':
        from models.densenet import densenet169
        net = densenet169(num_class=args.num_class)
    elif args.net == 'densenet201':
        from models.densenet import densenet201
        net = densenet201(num_class=args.num_class)
    elif args.net == 'googlenet':
        from models.googlenet import googlenet
        net = googlenet(num_class=args.num_class)
    elif args.net == 'inceptionv3':
        from models.inceptionv3 import inceptionv3
        net = inceptionv3(num_class=args.num_class)
    elif args.net == 'inceptionv4':
        from models.inceptionv4 import inceptionv4
        net = inceptionv4(num_class=args.num_class)
    elif args.net == 'inceptionresnetv2':
        from models.inceptionv4 import inception_resnet_v2
        net = inception_resnet_v2(num_class=args.num_class)
    elif args.net == 'xception':
        from models.xception import xception
        net = xception(num_class=args.num_class)
    elif args.net == 'resnet18':
        from models.resnet import resnet18
        net = resnet18(num_class=args.num_class)
    elif args.net == 'resnet34':
        from models.resnet import resnet34
        net = resnet34(num_class=args.num_class)
    elif args.net == 'resnet50':
        from models.resnet import resnet50
        net = resnet50(num_class=args.num_class)
    elif args.net == 'resnet101':
        from models.resnet import resnet101
        net = resnet101(num_class=args.num_class)
    elif args.net == 'resnet152':
        from models.resnet import resnet152
        net = resnet152(num_class=args.num_class)
    elif args.net == 'preactresnet18':
        from models.preactresnet import preactresnet18
        net = preactresnet18(num_class=args.num_class)
    elif args.net == 'preactresnet34':
        from models.preactresnet import preactresnet34
        net = preactresnet34(num_class=args.num_class)
    elif args.net == 'preactresnet50':
        from models.preactresnet import preactresnet50
        net = preactresnet50(num_class=args.num_class)
    elif args.net == 'preactresnet101':
        from models.preactresnet import preactresnet101
        net = preactresnet101(num_class=args.num_class)
    elif args.net == 'preactresnet152':
        from models.preactresnet import preactresnet152
        net = preactresnet152(num_class=args.num_class)
    elif args.net == 'resnext50':
        from models.resnext import resnext50
        net = resnext50(num_class=args.num_class)
    elif args.net == 'resnext101':
        from models.resnext import resnext101
        net = resnext101(num_class=args.num_class)
    elif args.net == 'resnext152':
        from models.resnext import resnext152
        net = resnext152(num_class=args.num_class)
    elif args.net == 'shufflenet':
        from models.shufflenet import shufflenet
        net = shufflenet(num_class=args.num_class)
    elif args.net == 'shufflenetv2':
        from models.shufflenetv2 import shufflenetv2
        net = shufflenetv2(num_class=args.num_class)
    elif args.net == 'squeezenet':
        from models.squeezenet import squeezenet
        net = squeezenet(num_class=args.num_class)
    elif args.net == 'mobilenet':
        from models.mobilenet import mobilenet
        net = mobilenet(num_class=args.num_class)
    elif args.net == 'mobilenetv2':
        from models.mobilenetv2 import mobilenetv2
        net = mobilenetv2(num_class=args.num_class)
    elif args.net == 'nasnet':
        from models.nasnet import nasnet
        net = nasnet(num_class=args.num_class)
    elif args.net == 'attention56':
        from models.attention import attention56
        net = attention56(num_class=args.num_class)
    elif args.net == 'attention92':
        from models.attention import attention92
        net = attention92(num_class=args.num_class)
    elif args.net == 'seresnet18':
        from models.senet import seresnet18
        net = seresnet18(num_class=args.num_class)
    elif args.net == 'seresnet34':
        from models.senet import seresnet34
        net = seresnet34(num_class=args.num_class)
    elif args.net == 'seresnet50':
        from models.senet import seresnet50
        net = seresnet50(num_class=args.num_class)
    elif args.net == 'seresnet101':
        from models.senet import seresnet101
        net = seresnet101(num_class=args.num_class)
    elif args.net == 'seresnet152':
        from models.senet import seresnet152
        net = seresnet152(num_class=args.num_class)

    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    if args.gpu:  # use_gpu
        net = net.to(settings.CUDA)

    return net


def get_training_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True, dataset_name='cifar100'):
    """ return training dataloader
    Args:
        mean: mean of training dataset
        std: std of training dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: training_loader: torch dataloader object
    """
    transform_train = None
    if dataset_name == 'cifar100' or dataset_name == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    elif dataset_name == 'tiny_imagenet':
        transform_train = transforms.Compose([
            transforms.RandomCrop(64, padding=8),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    elif dataset_name == 'Caltech256':
        transform_train = transforms.Compose([
            transforms.Resize(256),  # Resizes short size of the PIL image to 256
            transforms.CenterCrop(224),  # Crops a central square patch of the image 224 because torchvision's AlexNet needs a 224x224 input! Remember this when applying different transformations, otherwise you get an error
            transforms.ToTensor(),  # Turn PIL Image to torch.Tensor
            transforms.Normalize(mean, std)  # Normalizes tensor with mean and standard deviation
        ])

    if dataset_name == 'cifar100':
        training_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    elif dataset_name == 'cifar10':
        training_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    elif dataset_name == 'tiny_imagenet':
        training_dataset = TinyImagenet200(root='./data', train=True, download=True, transform=transform_train)
    elif dataset_name == 'caltech256':
        all_dataset = torchvision.datasets.Caltech256(root='./data', download=False, transform=transform_train)
        indices = list(range(len(all_dataset)))
        test_indices = random.sample(range(0, len(all_dataset)), int(len(all_dataset) * 0.2))
        train_indices = [i for i in indices if i not in test_indices]

        training_loader = DataLoader(Subset(all_dataset, torch.tensor(train_indices)), shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
        testing_loader = DataLoader(Subset(all_dataset, torch.tensor(test_indices)), shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
        return training_loader, testing_loader
    else:
        training_dataset = []

    training_loader = DataLoader(training_dataset, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
    return training_loader


def get_test_dataloader_all(mean, std, batch_size=16, num_workers=2, shuffle=True, dataset_name='cifar100'):
    """ return training dataloader
    Args:
        mean: mean of test dataset
        std: std of test dataset
        path: path to test python dataset
        batch_size: batchsize
        num_workers: num_works
        shuffle: whether to shuffle
    Returns: test_loader: torch dataloader object
    """

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    # cifar100_test = CIFAR100Test(path, transform=transform_test)
    if dataset_name == 'cifar100':
        test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True,
                                                     transform=transform_test)
    elif dataset_name == 'cifar10':
        test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    elif dataset_name == 'tiny_imagenet':
        test_dataset = TinyImagenet200(root='./data', train=False, download=True, transform=transform_test)
    else:
        test_dataset = []

    test_loader = DataLoader(test_dataset, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return test_loader


def sampling_training_data_loader(mean, std, num_sub_models=5, sample_proportion=0.2, batch_size=128, num_workers=2, shuffle=True, dataset_name='cifar100'):
    '''
    Return the randomly sampled training dataset, including the left subset,
    the blunt subset, and the label that whether it is a blunt sample
    :param mean: the mean of dataset
    :param std: the standard variation of dataset
    :param num_sub_models: the number of sub-models and sampling times
    :param sample_proportion: the sampling proportion for each sub-model
    :param batch_size: the batch size for each subset
    :param num_workers: the number_workers for each subset
    :param shuffle: whether shuffle the dataset
    :param dataset_name: dataset name to be sampled
    :return:
        the left sub-dataset list
        the blunt sub-dataset list
        the training dataset loader
        the label list that whether it is a blunt sample for each subset, 0 stands for left, 1 stands for blunt
    '''
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    if dataset_name == 'cifar100':
        training_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True,
                                                         transform=transform_train)
    elif dataset_name == 'cifar10':
        training_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                                        transform=transform_train)
    else:
        training_dataset = []

    indices = list(range(len(training_dataset)))
    blunt_label_list = []

    blunt_data_loader = []
    left_data_loader = []

    for i in range(num_sub_models):
        # sample the blunt indices by sampling proportion
        blunt_indices = random.sample(range(0, len(training_dataset)), int(len(training_dataset) * sample_proportion))

        # get the left indices
        left_indices = [index for index in indices if index not in blunt_indices]

        # label the blunt list
        blunt_label_temp = [False] * len(training_dataset)
        for index in blunt_indices:
            blunt_label_temp[index] = True

        blunt_label_list.append(blunt_label_temp)

        # keep the blunt data loader and left data loader
        blunt_data_loader.append(DataLoader(Subset(training_dataset, torch.tensor(blunt_indices)), shuffle=shuffle, num_workers=num_workers, batch_size=batch_size))
        left_data_loader.append(DataLoader(Subset(training_dataset, torch.tensor(left_indices)), shuffle=shuffle, num_workers=num_workers, batch_size=batch_size))

    blunt_label_list = torch.tensor(blunt_label_list)
    # should not be shuffled, because it should be consistent to the blunt label list
    # and used to fixing the original model later
    training_loader = DataLoader(training_dataset, shuffle=False, num_workers=num_workers, batch_size=batch_size)
    return training_loader, blunt_data_loader, left_data_loader, blunt_label_list


def partition_training_data_loader(mean, std, num_sub_models=5, batch_size=128, num_workers=2, shuffle=True, dataset_name='cifar100'):
    '''
    Return the randomly partitioned training dataset, including the left subset,
    the blunt subset, and the label that whether it is a blunt sample
    :param mean: the mean of dataset
    :param std: the standard variation of dataset
    :param num_sub_models: the number of sub-models and sampling times
    :param batch_size: the batch size for each subset
    :param num_workers: the number_workers for each subset
    :param shuffle: whether shuffle the dataset
    :param dataset_name: dataset name to be sampled
    :return:
        the left sub-dataset list
        the blunt sub-dataset list
        the training dataset loader
        the label list that whether it is a blunt sample for each subset, 0 stands for left, 1 stands for blunt
    '''
    transform_train = None
    if dataset_name == 'cifar100' or dataset_name == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    elif dataset_name == 'tiny_imagenet':
        transform_train = transforms.Compose([
            transforms.RandomCrop(64, padding=8),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    if dataset_name == 'cifar100':
        training_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    elif dataset_name == 'cifar10':
        training_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    elif dataset_name == 'tiny_imagenet':
        training_dataset = TinyImagenet200(root='./data', train=True, download=True, transform=transform_train)
    else:
        training_dataset = []

    indices = list(range(len(training_dataset)))
    random.shuffle(indices)
    blunt_label_list = []

    blunt_data_loader = []
    left_data_loader = []

    for i in range(num_sub_models):
        # partition the blunt indices by sampling proportion
        blunt_indices = indices[int(len(training_dataset) * i / num_sub_models):int(len(training_dataset) * (i + 1) / num_sub_models)]

        # get the left indices
        left_indices = [index for index in indices if index not in blunt_indices]

        # label the blunt list
        blunt_label_temp = [False] * len(training_dataset)
        for index in blunt_indices:
            blunt_label_temp[index] = True

        blunt_label_list.append(blunt_label_temp)

        # keep the blunt data loader and left data loader
        blunt_data_loader.append(DataLoader(Subset(training_dataset, torch.tensor(blunt_indices)), shuffle=shuffle, num_workers=num_workers, batch_size=batch_size))
        left_data_loader.append(DataLoader(Subset(training_dataset, torch.tensor(left_indices)), shuffle=shuffle, num_workers=num_workers, batch_size=batch_size))

    blunt_label_list = torch.tensor(blunt_label_list)
    # should not be shuffled, because it should be consistent to the blunt label list
    # and used to fixing the original model later
    training_loader = DataLoader(training_dataset, shuffle=False, num_workers=num_workers, batch_size=batch_size)
    return training_loader, blunt_data_loader, left_data_loader, blunt_label_list


def original_model_failed_test_subset(model_args, model, mean, std, batch_size=16, num_workers=2, shuffle=True, dataset_name='cifar100'):
    '''
    Get the test subset that failed in the original model
    :param model_args:
    :param model:
    :param mean:
    :param std:
    :param batch_size:
    :param num_workers:
    :param shuffle:
    :param dataset_name:
    :return: the test subset that failed in the original model
    '''
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    # cifar100_test = CIFAR100Test(path, transform=transform_test)
    if dataset_name == 'cifar100':
        test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True,
                                                     transform=transform_test)
    elif dataset_name == 'cifar10':
        test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    elif dataset_name == 'tiny_imagenet':
        test_dataset = TinyImagenet200(root='./data', train=False, download=True, transform=transform_test)
    else:
        test_dataset = []

    model.eval()
    failed_index = []
    current_len = 0
    with torch.no_grad():
        test_loader = DataLoader(test_dataset, shuffle=False, num_workers=num_workers, batch_size=128)
        for index, (image, label) in enumerate(test_loader):
            if model_args.gpu:
                image = image.to(settings.CUDA)
                label = label.to(settings.CUDA)

            outputs = model(image)
            _, predicts = outputs.max(1)
            result = predicts.eq(label)
            for i in range(len(result)):
                if not result[i]:
                    failed_index.append(current_len + i)

            current_len += len(image)
            del image, label
            if model_args.gpu:
                with torch.cuda.device(settings.CUDA):
                    torch.cuda.empty_cache()

    test_loader = DataLoader(Subset(test_dataset, torch.tensor(failed_index)), shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
    return test_loader


def get_voting_training_dataloader(mean, std, equal_divide_n=3, batch_size=16, num_workers=2, shuffle=True, dataset_name='cifar100'):
    """ return training dataloader
    Args:
        mean: mean of training dataset
        std: std of training dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: training_loader: torch dataloader object
    """
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    if dataset_name == 'cifar100':
        training_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True,
                                                         transform=transform_train)
    elif dataset_name == 'cifar10':
        training_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                                        transform=transform_train)
    else:
        training_dataset = []

    indices = list(range(len(training_dataset)))
    random.shuffle(indices)

    equal_divide_indices = []  # 1/n indices should be unlearned
    left_indices = []  # (n-1)/n indices should be left

    equal_divide_dataloader = []  # 1/n dataset should be unlearned
    left_dataloader = []  # (n-1)/n dataset should be left

    for i in range(equal_divide_n):
        equal_divide_indices.append(torch.tensor(indices[int(len(training_dataset) * i / equal_divide_n): int(
            len(training_dataset) * (i + 1) / equal_divide_n)]))
        equal_divide_dataloader.append(
            DataLoader(Subset(training_dataset, equal_divide_indices[i]), shuffle=shuffle, num_workers=num_workers,
                       batch_size=batch_size))

        left_indices.append(torch.tensor([j for j in indices if j not in equal_divide_indices[i]]))
        left_dataloader.append(
            DataLoader(Subset(training_dataset, left_indices[i]), shuffle=shuffle, num_workers=num_workers,
                       batch_size=batch_size))

    return equal_divide_dataloader, left_dataloader


def get_confidence_prediction(model_args, model, data_loader, batch_size=16, num_workers=2, shuffle=True):
    '''
    test the model on data and return the prediction result and the confidence list in ground truth
    :param shuffle:
    :param batch_size:
    :param num_workers:
    :param data_loader:
    :param model_args:
    :param model:
    :param datal_oader:
    :return: the prediction result and the confidence list in ground truth
    '''
    sf = torch.nn.Softmax(dim=1)

    model.eval()
    confidence_list = None
    prediction_list = None
    ground_truth = None

    with torch.no_grad():
        for images, labels in data_loader:
            if model_args.gpu:
                images = images.to(settings.CUDA)
                labels = labels.to(settings.CUDA)

            outputs = sf(model(images))
            # confidence, predicts = outputs.max(1)
            confidence = torch.tensor([outputs[i][labels[i]] for i in range(len(labels))])
            _, predicts = outputs.max(1)

            if confidence_list is None:
                confidence_list = confidence
            else:
                confidence_list = torch.cat((confidence_list, confidence))

            if prediction_list is None:
                prediction_list = predicts.eq(labels)
            else:
                prediction_list = torch.cat((prediction_list, predicts.eq(labels)))

            if ground_truth is None:
                ground_truth = labels
            else:
                ground_truth = torch.cat((ground_truth, labels))

            del images, labels
            if model_args.gpu:
                with torch.cuda.device(settings.CUDA):
                    torch.cuda.empty_cache()

    average_confidence = torch.mean(confidence_list)
    high_pass_list = torch.tensor([confidence_list[i] for i in range(len(data_loader.dataset)) if
                                   confidence_list[i] >= average_confidence and prediction_list[i] == True])
    high_fail_list = torch.tensor([confidence_list[i] for i in range(len(data_loader.dataset)) if
                                   confidence_list[i] >= average_confidence and prediction_list[i] == False])
    low_pass_list = torch.tensor([confidence_list[i] for i in range(len(data_loader.dataset)) if
                                  confidence_list[i] < average_confidence and prediction_list[i] == True])
    low_fail_list = torch.tensor([confidence_list[i] for i in range(len(data_loader.dataset)) if
                                  confidence_list[i] < average_confidence and prediction_list[i] == False])

    good_index = [i for i in range(len(data_loader.dataset)) if
                  confidence_list[i] > average_confidence and prediction_list[i] == True]
    bad_index = [i for i in range(len(data_loader.dataset)) if i not in good_index]

    bad_class = torch.tensor([ground_truth[i] for i in bad_index])
    bad_data_loader = DataLoader(Subset(data_loader.dataset, torch.tensor(bad_index)), shuffle=shuffle,
                                 num_workers=num_workers, batch_size=batch_size)
    good_data_loader = DataLoader(Subset(data_loader.dataset, torch.tensor(good_index)), shuffle=shuffle,
                                  num_workers=num_workers, batch_size=batch_size)

    # from matplotlib import pyplot as plt
    # bins = np.linspace(0, 99, 100)
    # plt.hist(bad_class, bins)
    # plt.savefig("checkpoint/mobilenet/bad_class.png")
    # plt.close()

    # high_pass_list.sort(reverse=True)
    # high_fail_list.sort(reverse=True)
    # low_pass_list.sort(reverse=True)
    # low_fail_list.sort(reverse=True)
    # from matplotlib import pyplot as plt
    # plt.title('Confidence List')
    # plt.plot(range(len(high_pass_list)), high_pass_list, label='high pass')
    # plt.plot(range(len(high_pass_list), len(high_pass_list)+len(high_fail_list)), high_fail_list, label='high fail')
    # plt.plot(range(len(high_pass_list)+len(high_fail_list), len(high_pass_list)+len(high_fail_list)+len(low_pass_list)), low_pass_list, label='low pass')
    # plt.plot(range(len(high_pass_list)+len(high_fail_list)+len(low_pass_list), len(high_pass_list)+len(high_fail_list)+len(low_pass_list)+len(low_fail_list)), low_fail_list, label='low fail')
    # plt.legend()
    # plt.savefig("checkpoint/mobilenet/confidence.png")
    # plt.close()
    return bad_data_loader, good_data_loader, prediction_list, (
    confidence_list, high_pass_list, high_fail_list, low_pass_list, low_fail_list), bad_class


def get_voting_training_dataloader_by_class(bad_index, mean, std, batch_size=16, num_workers=2, shuffle=True, dataset_name='cifar100', train=True):
    """ return deleted and left training dataloader by class indexes list
    Args:
        mean: mean of training dataset
        std: std of training dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: training_loader: torch dataloader object
    """

    if train:
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    # cifar100_training = CIFAR100Train(path, transform=transform_train)
    if dataset_name == 'cifar100':
        training_dataset = torchvision.datasets.CIFAR100(root='./data', train=train, download=True, transform=transform)
    elif dataset_name == 'cifar10':
        training_dataset = torchvision.datasets.CIFAR10(root='./data', train=train, download=True, transform=transform)
    else:
        training_dataset = []

    targets = list(training_dataset.targets)

    delete_sample_indices = torch.tensor([j for j in range(len(targets)) if targets[j] in bad_index])
    left_sample_indices = torch.tensor([j for j in range(len(targets)) if targets[j] not in bad_index])

    bad_dataloader = DataLoader(Subset(training_dataset, delete_sample_indices), shuffle=shuffle,
                                num_workers=num_workers, batch_size=batch_size)
    good_dataloader = DataLoader(Subset(training_dataset, left_sample_indices), shuffle=shuffle,
                                 num_workers=num_workers, batch_size=batch_size)

    return bad_dataloader, good_dataloader


def get_voting_training_dataloader_by_class_random(mean, std, equal_divide_n=3, batch_size=16, num_workers=2, shuffle=True, dataset_name='cifar100'):
    """ return training dataloader
    Args:
        mean: mean of training dataset
        std: std of training dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: training_loader: torch dataloader object
    """
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # cifar100_training = CIFAR100Train(path, transform=transform_train)
    if dataset_name == 'cifar100':
        training_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True,
                                                         transform=transform_train)
        class_num = 100
    elif dataset_name == 'cifar10':
        training_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                                        transform=transform_train)
        class_num = 10
    else:
        training_dataset = []
        class_num = 0

    class_indices = list(range(class_num))
    random.shuffle(class_indices)
    equal_divide_indices = []  # 100/n classes should be unlearned

    deleted_dataloader = []  # 100/n*len(dataset) samples should be unlearned
    left_dataloader = []  # 100*(n-1)/n*len(dataset) samples should be left

    targets = list(training_dataset.targets)

    for i in range(equal_divide_n):
        equal_divide_indices.append(
            class_indices[int(class_num * i / equal_divide_n):int(class_num * (i + 1) / equal_divide_n)])
        delete_sample_indices = torch.tensor([j for j in range(len(targets)) if targets[j] in equal_divide_indices[i]])
        left_sample_indices = torch.tensor(
            [j for j in range(len(targets)) if targets[j] not in equal_divide_indices[i]])

        deleted_dataloader.append(
            DataLoader(Subset(training_dataset, delete_sample_indices), shuffle=shuffle, num_workers=num_workers,
                       batch_size=batch_size))
        left_dataloader.append(
            DataLoader(Subset(training_dataset, left_sample_indices), shuffle=shuffle, num_workers=num_workers,
                       batch_size=batch_size))

    return deleted_dataloader, left_dataloader


def get_test_dataloader(mean, std, batch_size=16, shuffle=False, dataset_name='cifar100'):
    """ return valid and test dataloader
    Args:
        mean: mean of test dataset
        std: std of test dataset
        batch_size: batchsize
        shuffle: whether to shuffle
    Returns: test_loader: torch dataloader object
    """

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    # cifar100_test = CIFAR100Test(path, transform=transform_test)
    if dataset_name == 'cifar100':
        test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True,
                                                     transform=transform_test)
    elif dataset_name == 'cifar10':
        test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    else:
        test_dataset = []

    # cifar100_test_loader = DataLoader(cifar100_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    train_validation_size = int(0.5 * len(test_dataset))
    test_size = len(test_dataset) - train_validation_size
    train_validation_dataset, test_dataset = random_split(test_dataset, [train_validation_size, test_size],
                                                          generator=torch.Generator().manual_seed(8842))

    valid_dataset_loader = DataLoader(train_validation_dataset, shuffle=shuffle, batch_size=batch_size)
    test_dataset_loader = DataLoader(test_dataset, shuffle=shuffle, batch_size=batch_size)

    return valid_dataset_loader, test_dataset_loader


# def get_retraining_dataloader(mean, std, delete_ratio=0.1, batch_size=16, num_workers=2, shuffle=True, dataset_name='cifar100'):
#     """ return retraining dataloader
#     Args:
#         mean: mean of cifar100 training dataset
#         std: std of cifar100 training dataset
#         retrain_ratio: the percentage of full data used for retraining, default is 0.9
#         path: path to cifar100 training python dataset
#         batch_size: dataloader batchsize
#         num_workers: dataloader num_works
#         shuffle: whether to shuffle
#     Returns: train_data_loader:torch dataloader object
#     """
#     transform_train = transforms.Compose([
#         transforms.RandomCrop(32, padding=4),
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomRotation(15),
#         transforms.ToTensor(),
#         transforms.Normalize(mean, std)
#     ])
#
#     if dataset_name == 'cifar100':
#         full_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
#     elif dataset_name == 'cifar10':
#         full_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
#
#     # # Creating data indices for training and validation splits:
#     # indices = list(range(len(cifar100_training)))
#     # split = int(len(cifar100_training) - len(cifar100_training) * retrain_ratio)
#     # left_indices, deleted_indices = indices[split:], indices[:split]
#     #
#     # # Creating PT data samplers and loaders:
#     # left_sampler = SubsetRandomSampler(left_indices)
#     # deleted_sampler = SubsetRandomSampler(deleted_indices)
#     #
#     # cifar100_training_left_loader = DataLoader(cifar100_training, batch_size=batch_size, sampler=left_sampler)
#     # cifar100_training_deleted_loader = DataLoader(cifar100_training, batch_size=batch_size, sampler=deleted_sampler)
#
#     delete_size = int(delete_ratio * len(full_dataset))
#     left_size = len(full_dataset) - delete_size
#     left_dataset, delete_dataset = random_split(full_dataset, [left_size, delete_size])
#
#     left_dataset_loader = DataLoader(left_dataset,  shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
#     deleted_dataset_loader = DataLoader(delete_dataset,  shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
#
#     return left_dataset_loader, deleted_dataset_loader


def get_retraining_dataloader(dataset_loader, delete_num=100, batch_size=16, num_workers=2, shuffle=True):
    """ return retraining dataloader
    Args:
        mean: mean of cifar100 training dataset
        std: std of cifar100 training dataset
        retrain_ratio: the percentage of full data used for retraining, default is 0.9
        path: path to cifar100 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: train_data_loader:torch dataloader object
    """

    left_num = len(dataset_loader.dataset) - delete_num
    left_dataset, delete_dataset = random_split(dataset_loader.dataset, [left_num, delete_num])

    left_dataset_loader = DataLoader(left_dataset, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
    deleted_dataset_loader = DataLoader(delete_dataset, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return left_dataset_loader, deleted_dataset_loader


# def get_retraining_one_class_dataloader(mean, std, delete_num=100, delete_class=1, batch_size=16, num_workers=2, shuffle=True, dataset_name='cifar100'):
#     """ return retraining dataloader
#     Args:
#         mean: mean of training dataset
#         std: std of training dataset
#         retrain_ratio: the percentage of full data used for retraining, default is 0.9
#         path: path to training python dataset
#         batch_size: dataloader batchsize
#         num_workers: dataloader num_works
#         shuffle: whether to shuffle
#     Returns: train_data_loader:torch dataloader object
#     """
#     transform_train = transforms.Compose([
#         transforms.RandomCrop(32, padding=4),
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomRotation(15),
#         transforms.ToTensor(),
#         transforms.Normalize(mean, std)
#     ])
#
#     if dataset_name == 'cifar100':
#         full_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
#     elif dataset_name == 'cifar10':
#         full_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
#     else:
#         full_dataset = []
#
#     deleted_indices = np.random.choice(np.where(np.array(full_dataset.targets) == delete_class)[0], size=delete_num, replace=False)
#     left_indices = [i for i in range(len(full_dataset)) if i not in deleted_indices]
#
#     # Creating PT data samplers and loaders:
#     left_sampler = SubsetRandomSampler(left_indices)
#     deleted_sampler = SubsetRandomSampler(deleted_indices)
#
#     left_dataset_loader = DataLoader(full_dataset, batch_size=batch_size, sampler=left_sampler)
#     deleted_dataset_loader = DataLoader(full_dataset, batch_size=1, sampler=deleted_sampler)
#
#     return left_dataset_loader, deleted_dataset_loader


def get_one_class_other_dataloader(mean, std, train_class=0, train=True, batch_size=16, num_workers=2, shuffle=True, dataset_name='cifar100'):
    """ return training dataloader
    Args:
        mean: mean of training dataset
        std: std of training dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: training_loader: torch dataloader object
    """
    if train and dataset_name == 'cifar100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    # cifar100_training = CIFAR100Train(path, transform=transform_train)
    if dataset_name == 'cifar100':
        training_dataset = torchvision.datasets.CIFAR100(root='./data', train=train, download=True,
                                                         transform=transform_train)
    elif dataset_name == 'cifar10':
        training_dataset = torchvision.datasets.CIFAR10(root='./data', train=train, download=True,
                                                        transform=transform_train)
    elif dataset_name == 'mnist':
        training_dataset = torchvision.datasets.MNIST(root='./data', train=train, download=True,
                                                      transform=transform_train)
    else:
        training_dataset = []

    # train_indices = torch.tensor(np.where(np.array(training_dataset.targets) <= train_class)[0])
    train_indices = torch.tensor(np.where(np.array(training_dataset.targets) == train_class)[0])
    # other_indices = torch.tensor(np.where(np.array(training_dataset.targets) > train_class)[0])
    other_indices = torch.tensor(np.where(np.array(training_dataset.targets) == train_class + 1)[0])

    train_dataset = Subset(training_dataset, train_indices)
    train_dataset_loader = DataLoader(train_dataset, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    if len(other_indices) > 0:
        other_dataset = Subset(training_dataset, other_indices)
        other_dataset_loader = DataLoader(other_dataset, shuffle=shuffle, num_workers=num_workers,
                                          batch_size=batch_size)
    else:
        other_dataset_loader = []

    return train_dataset_loader, other_dataset_loader


def get_target_shadow_train_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True, dataset_name='cifar100', dataset_size=10000):
    """ return target model training dataset and shadow model dataset dataloader
    Args:
        mean: mean of training dataset
        std: std of training dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns:
        target_train_dataset_loader: torch dataloader object
        shadow_dataset_loader: torch dataloader object
    """

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    # cifar100_training = CIFAR100Train(path, transform=transform_train)
    if dataset_name == 'cifar100':
        training_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True,
                                                         transform=transform_train)
    elif dataset_name == 'cifar10':
        training_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                                        transform=transform_train)

    # training_loader = DataLoader(training_dataset, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    target_train_size = dataset_size
    shadow_size = len(training_dataset) - target_train_size
    target_train_dataset, shadow_dataset = random_split(training_dataset, [target_train_size, shadow_size])

    target_train_dataset_loader = DataLoader(target_train_dataset, shuffle=shuffle, num_workers=num_workers,
                                             batch_size=batch_size)

    return target_train_dataset_loader, shadow_dataset


def get_shadow_dataloader(dataset, batch_size=16, num_workers=2, shuffle=True, data_size=10000):
    """ return shadow model training and testing dataloader
    Args:
        shadow_dataset_loader: the dataset loader of shadow dataset
        batch_size: batchsize
        num_workers: num_works
        shuffle: whether to shuffle
    Returns:
        shadow_train_dataset_loader: torch dataloader object
        shadow_test_dataset_loader: torch dataloader object
    """

    shadow_train_size = data_size
    shadow_test_size = data_size
    left_size = len(dataset) - 2 * data_size
    shadow_dataset, left_dataset = random_split(dataset, [data_size * 2, left_size])
    shadow_train_dataset, shadow_test_dataset = random_split(shadow_dataset, [shadow_train_size, shadow_test_size])

    shadow_train_dataset_loader = DataLoader(shadow_train_dataset, shuffle=shuffle, num_workers=num_workers,
                                             batch_size=batch_size)
    shadow_test_dataset_loader = DataLoader(shadow_test_dataset, shuffle=shuffle, num_workers=num_workers,
                                            batch_size=batch_size)

    return shadow_train_dataset_loader, shadow_test_dataset_loader


def compute_mean_std(cifar100_dataset):
    """compute the mean and std of cifar100 dataset
    Args:
        cifar100_training_dataset or cifar100_test_dataset
        witch derived from class torch.utils.data

    Returns:
        a tuple contains mean, std value of entire dataset
    """

    data_r = numpy.dstack([cifar100_dataset[i][1][:, :, 0] for i in range(len(cifar100_dataset))])
    data_g = numpy.dstack([cifar100_dataset[i][1][:, :, 1] for i in range(len(cifar100_dataset))])
    data_b = numpy.dstack([cifar100_dataset[i][1][:, :, 2] for i in range(len(cifar100_dataset))])
    mean = numpy.mean(data_r), numpy.mean(data_g), numpy.mean(data_b)
    std = numpy.std(data_r), numpy.std(data_g), numpy.std(data_b)

    return mean, std


class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """

    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


class NewDataSet(Dataset):
    def __init__(self, images, labels, transform=None, target_transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return len(self.images)


def train_one_epoch(model_args, model, train_loader, optimizer, loss_function, sample_weight=None):
    model.train()
    train_loss_temp = 0.0
    correct = 0

    for index, (images, labels) in enumerate(train_loader):
        if model_args.gpu:
            images = images.to(settings.CUDA)
            labels = labels.to(settings.CUDA)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_function(outputs, labels)
        train_loss_temp += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

        loss.backward()
        optimizer.step()

        del images, labels
        if model_args.gpu:
            with torch.cuda.device(settings.CUDA):
                torch.cuda.empty_cache()

    return float(correct) / len(train_loader.dataset), train_loss_temp


@torch.no_grad()
def test_model(model_args, model, test_loader, loss_function):
    model.eval()

    test_loss = 0.0  # cost function error
    correct = 0

    with torch.no_grad():
        for images, labels in test_loader:
            if model_args.gpu:
                images = images.to(settings.CUDA)
                labels = labels.to(settings.CUDA)

            outputs = model(images)
            loss = loss_function(outputs, labels)
            test_loss += loss.item()
            _, predicts = outputs.max(1)
            correct += predicts.eq(labels).sum()

            del images, labels
            if model_args.gpu:
                with torch.cuda.device(settings.CUDA):
                    torch.cuda.empty_cache()

    return float(correct) / len(test_loader.dataset), test_loss


def test_each_class(model_args, model, loss_function, train=False):
    if model_args.dataset_name == 'cifar100':
        mean = settings.CIFAR100_TRAIN_MEAN
        std = settings.CIFAR100_TRAIN_STD
    elif model_args.dataset_name == 'tiny_imagenet':
        mean = settings.TINY_IMAGENET_MEAN
        std = settings.TINY_IMAGENET_STD
    else:
        mean = settings.CIFAR10_TRAIN_MEAN
        std = settings.CIFAR10_TRAIN_STD

    if train:
        all_dataloader = get_training_dataloader(mean, std, num_workers=4, batch_size=model_args.b, shuffle=True, dataset_name=model_args.dataset_name)
    else:
        all_dataloader = get_test_dataloader_all(mean, std, num_workers=4, batch_size=model_args.b, shuffle=True, dataset_name=model_args.dataset_name)

    mean_test_acc, _ = test_model(model_args=model_args, model=model, test_loader=all_dataloader, loss_function=loss_function)

    test_acc_list_all_class = []
    for i in range(model_args.num_class):
        test_loader, _ = get_one_class_other_dataloader(
            settings.CIFAR100_TRAIN_MEAN,
            settings.CIFAR100_TRAIN_STD,
            train_class=i,
            train=train,
            batch_size=model_args.b,
            num_workers=4,
            shuffle=True,
            dataset_name=model_args.dataset_name
        )

        test_acc, _ = test_model(model_args=model_args, model=model, test_loader=test_loader,
                                 loss_function=loss_function)
        test_acc_list_all_class.append(test_acc)

    # lower_mean_acc = list(np.where(np.array(test_acc_list_all_class) < mean_test_acc)[0])
    # return lower_mean_acc
    return test_acc_list_all_class

@torch.no_grad()
def voting(model_args, models, test_loader):
    vote_correct = 0
    all_unreached = 0  # all sub-models cannot reach the agreement
    first_two_unreached = 0  # the first two voting results are equal

    with torch.no_grad():
        for images, labels in test_loader:
            if model_args.gpu:
                images = images.to(settings.CUDA)

            outputs = []
            for index in range(len(models)):
                models[index].eval()
                output = models[index](images)
                _, preds = output.max(1)
                outputs.append(preds.tolist())

            vote_result = []
            for j in range(len(images)):
                # vote_result.append(np.argmax(np.bincount(np.array(outputs)[:, j])))
                counts_array = np.bincount(np.array(outputs)[:, j])
                max_index = np.argmax(counts_array)
                vote_result.append(max_index)
                if counts_array[max_index] == 1:  # count the number with unreached voting result
                    all_unreached += 1
                else:
                    max_votes = counts_array[max_index]
                    counts_array[max_index] = 0
                    second_max_index = np.argmax(counts_array)
                    if max_votes == counts_array[second_max_index]:  # count the first two voting results are equal
                        first_two_unreached += 1

            vote_correct += sum(np.array(labels.tolist()) == np.array(vote_result))

            del images, labels
            if model_args.gpu:
                with torch.cuda.device(settings.CUDA):
                    torch.cuda.empty_cache()

    return float(vote_correct) / len(test_loader.dataset), all_unreached, first_two_unreached


class UniformNormLossFunc(torch.nn.Module):
    def __init__(self):
        super(UniformNormLossFunc, self).__init__()

    def forward(self, inputs, targets):
        # L1-norm distance
        # distance = torch.norm(inputs - targets, p=1)

        # l2-norm distance
        distance = 0
        for index in range(len(inputs)):
            distance += torch.norm(inputs[index] - targets[index])

        return distance


class UniformNormLossFunc_2(torch.nn.Module):
    def __init__(self):
        super(UniformNormLossFunc_2, self).__init__()

    def forward(self, inputs, targets):
        # L1-norm distance
        # distance = torch.norm(inputs - targets, p=1)

        # l2-norm distance
        distance = 0
        for i in range(len(inputs[1])):
            distance_temp = 0
            for j in range(len(inputs[0])):
                distance_temp += torch.norm(inputs[j][i] - targets[i])

            distance += distance_temp / len(inputs[1])
        return distance


class WeightedCrossEntropyLossFunc(torch.nn.Module):
    def __init__(self, reduction='mean'):
        super(WeightedCrossEntropyLossFunc, self).__init__()
        self.reduction = reduction

    def forward(self, inputs, target, weight):
        exp = torch.exp(inputs)
        numerator = exp.gather(1, target.unsqueeze(-1)).squeeze()  # Numerator
        denominator = exp.sum(1)  # Denominator
        sf = weight * (numerator / denominator)  # ei / sum(ej)
        loss = -torch.log(sf)  # weight * -yi * log(pi)
        if self.reduction == "sum":
            return loss.sum()
        else:
            return loss.mean()

        # if logits.dim() > 2:
        #     logits = logits.view(logits.size(0), logits.size(1), -1)  # [N, C, HW]
        #     logits = logits.transpose(1, 2)   # [N, HW, C]
        #     logits = logits.contiguous().view(-1, logits.size(2))    # [NHW, C]
        # target = target.view(-1, 1)    # [NHW，1]
        #
        # logits = F.log_softmax(logits, 1)
        # logits = logits.gather(1, target)   # [NHW, 1]
        # loss = weight * (-1 * logits)
        #
        # if self.reduction == 'mean':
        #     loss = loss.mean()
        # elif self.reduction == 'sum':
        #     loss = loss.sum()
        # return loss


def adjustment_step(model_args, model, dataset_loader, loss_function, optimizer, blunt_strategy='uniform', test_acc_distribution_list=None):
    kl_loss_temp = 0.0
    sf = torch.nn.Softmax(dim=1)

    model.train()
    for images, labels in dataset_loader:
        blunt_tensor = []
        if blunt_strategy == 'uniform':
            blunt_tensor = sf(torch.full([len(images), model_args.num_class], 1 / model_args.num_class)).to(settings.CUDA)
        elif blunt_strategy == 'distribution' or blunt_strategy == 'random':
            for i in range(len(images)):
                blunt_tensor.append(test_acc_distribution_list)
            blunt_tensor = sf(torch.tensor(blunt_tensor))
        if model_args.gpu:
            images = images.to(settings.CUDA)
            blunt_tensor = blunt_tensor.to(settings.CUDA)

        optimizer.zero_grad()
        outputs = sf(model(images))
        loss = loss_function(outputs, blunt_tensor)
        kl_loss_temp += loss.item()
        loss.backward()
        optimizer.step()

    return kl_loss_temp


# evaluate target: uniform distribution
@torch.no_grad()
def eval_uniform_distribution(model_args, model, test_loader, n_classes, loss_function):
    model.eval()
    sf = torch.nn.Softmax(dim=1)

    # outputs = None
    kl_d = 0.0
    with torch.no_grad():
        for image, label in test_loader:
            uniform_tensor = sf(torch.full([len(image), n_classes], 1 / n_classes)).to(settings.CUDA)
            if model_args.gpu:
                image = image.to(settings.CUDA)
                uniform_tensor = uniform_tensor.to(settings.CUDA)

            outputs = sf(model(image))
            loss = loss_function(outputs, uniform_tensor)
            kl_d += loss.item()

    return kl_d


def models_delta_weights(state_dict_1, state_dict_2):
    '''
    Return the delta weights (state_dict_1 - state_dict_2)
    :param state_dict_1:
    :param state_dict_2:
    :return: delta weights of (state_dict_1 - state_dict_2)
    '''

    delta_weights = None
    for k, v in state_dict_2.items():
        if delta_weights is None:
            delta_weights = state_dict_1
        else:
            delta_weights[k] -= v

    return delta_weights


def models_add_delta_weights(state_dict_1, state_dict_2, learning_rate):
    '''
    Return the delta weights (state_dict_1 + state_dict_2)
    :param learning_rate:
    :param state_dict_1:
    :param state_dict_2:
    :return: delta weights of (state_dict_1 + state_dict_2)
    '''

    delta_weights = None
    for k, v in state_dict_2.items():
        if delta_weights is None:
            delta_weights = state_dict_1
        else:
            try:
                delta_weights[k] += learning_rate * v
            except:
                delta_weights[k] += torch.tensor(learning_rate, dtype=torch.long) * v

    return delta_weights


def models_average_weights(state_dict_1, state_dict_2):
    '''
    Return the delta weights (state_dict_1 + state_dict_2)
    :param learning_rate:
    :param state_dict_1:
    :param state_dict_2:
    :return: delta weights of (state_dict_1 + state_dict_2)
    '''

    delta_weights = None
    for k, v in state_dict_2.items():
        if delta_weights is None:
            delta_weights = state_dict_1
        else:
            try:
                delta_weights[k] += (delta_weights[k] + v) / 2
            except:
                delta_weights[k] += v

    return delta_weights


# un_use
@torch.no_grad()
def createExtraData(model_args, model, data_loader):
    model.eval()

    images_tensor = None
    labels_tensor = None
    with torch.no_grad():
        for (images, labels) in data_loader:
            if model_args.gpu:
                images_c = images.to(settings.CUDA)

            output = model(images_c)
            n_extra = 1
            _, prediction_smallest = output.topk(n_extra, 1, largest=False, sorted=True)

            for i in range(n_extra):
                if images_tensor is None:
                    images_tensor = images.clone()
                    labels_tensor = prediction_smallest[:, i].clone()
                else:
                    images_tensor = torch.cat((images_tensor, images))
                    labels_tensor = torch.cat((labels_tensor, prediction_smallest[:, i]))

    extra_dataset = NewDataSet(images_tensor, labels_tensor)
    extra_dataset_loader = DataLoader(extra_dataset, batch_size=128)
    return extra_dataset_loader


# un_use
@torch.no_grad()
def eval_difference(model_args, model, model_retrain, training_deleted_loader):
    model.eval()
    model_retrain.eval()
    counts = 0

    with torch.no_grad():
        for n_iter, (image, label) in enumerate(training_deleted_loader):
            if model_args.gpu:
                image = image.to(settings.CUDA)

            output = model(image)
            output2 = model_retrain(image)
            _, prediction = output.topk(1, 1, largest=True, sorted=True)
            _, prediction2 = output2.topk(1, 1, largest=True, sorted=True)
            for i in range(len(prediction2)):
                if prediction[i] == prediction2[i]:
                    counts += 1
        return counts / len(training_deleted_loader.dataset)
