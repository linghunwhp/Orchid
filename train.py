# train.py
# !/usr/bin/env	python3

import os
import argparse
import time
from copy import deepcopy
import torch
import torch.nn as nn
import torch.optim as optim
from conf import settings
from utils import get_network, get_training_dataloader, get_test_dataloader_all, train_one_epoch, test_model, get_voting_training_dataloader_by_class
from matplotlib import pyplot as plt


def train_model(model_args, train_loader, test_loader):
    model = get_network(model_args)
    loss_function = nn.CrossEntropyLoss()
    model_optimizer = optim.SGD(model.parameters(), lr=model_args.lr, momentum=0.9, weight_decay=1e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(model_optimizer, milestones=settings.MILESTONES, gamma=0.1)

    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, model_args.net, settings.TIME_NOW)
    # use tensorboard
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)

    # create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    best_train_acc = 0.0
    best_test_acc = 0.0
    best_epoch = 0
    best_model_state_dict = None

    train_acc_list = []
    train_loss_list = []
    test_acc_list = []
    test_loss_list = []

    main_process_start = time.time()
    for epoch in range(1, settings.EPOCH):
        if epoch > model_args.warm:
            train_scheduler.step()

        train_acc, train_loss = train_one_epoch(model_args=model_args, model=model, train_loader=train_loader, optimizer=model_optimizer, loss_function=loss_function)
        train_acc_list.append(train_acc)
        train_loss_list.append(train_loss)

        test_acc, test_loss = test_model(model_args=model_args, model=model, test_loader=test_loader, loss_function=loss_function)
        test_acc_list.append(test_acc)
        test_loss_list.append(test_loss)

        # start to save best performance model after learning rate decay to 0.01
        if epoch > settings.MILESTONES[0] and best_test_acc < test_acc:
            best_train_acc = train_acc
            best_test_acc = test_acc
            best_epoch = epoch
            best_model_state_dict = deepcopy(model.state_dict())
            torch.save(best_model_state_dict, checkpoint_path.format(net=model_args.net, epoch=best_epoch, type='best_test' + str(best_test_acc) + 'best_train' + str(best_train_acc)))

        if epoch > settings.MILESTONES[0] and epoch % 10 == 0:
            torch.save(model.state_dict(), checkpoint_path.format(net=model_args.net, epoch=epoch, type='best_test' + str(test_acc) + 'best_train' + str(train_acc)))

        print(f'Epoch: {epoch} Learning rate: {model_optimizer.param_groups[0]["lr"]:.4f} Train Loss: {train_loss:.4f} Test Loss: {test_loss:.4f} Train Acc: {train_acc:.4f} Test Acc: {test_acc:.4f}')

        if epoch % 10 == 0:
            plt.title('Train and Test Accuracy')
            plt.plot(range(1, len(train_acc_list)+1), train_acc_list, label='Train Accuracy')
            plt.plot(range(1, len(test_acc_list)+1), test_acc_list, label='Test Accuracy')
            plt.legend()
            plt.savefig(checkpoint_path.format(net=model_args.net, epoch=epoch, type="Accuracy_") + str(epoch) + "_.png")
            plt.close()

            plt.title('Train and Test Loss')
            plt.plot(range(1, len(train_loss_list)+1), train_loss_list, label='Train Loss')
            plt.plot(range(1, len(test_loss_list)+1), test_loss_list, label='Test Loss')
            plt.legend()
            plt.savefig(checkpoint_path.format(net=model_args.net, epoch=epoch, type="Loss_") + str(epoch) + "_.png")
            plt.close()

    # torch.save(best_model_state_dict, checkpoint_path.format(net=model_args.net, epoch=best_epoch, type='best_test' + str(best_test_acc) + 'best_train' + str(best_train_acc)))
    main_process_finish = time.time()
    print(f'Full training time consumed: {main_process_finish - main_process_start:.2f}s')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-b', type=int, default=128, help='batch size for data loader')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('-num_class', type=float, default=100, help='class number')
    parser.add_argument('-dataset_name', type=str, default='cifar100', help='dataset name')
    args = parser.parse_args()

    # data preprocessing:
    training_loader = get_training_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
        shuffle=True,
        dataset_name=args.dataset_name
    )

    testing_loader = get_test_dataloader_all(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
        shuffle=True,
        dataset_name=args.dataset_name
    )

    train_model(model_args=args, train_loader=training_loader, test_loader=testing_loader)

    # bad_set_index = [2, 3, 4, 10, 11, 13, 14, 16, 19, 22, 25, 26, 27, 29, 30, 32, 33, 35, 38, 40, 44, 45, 46, 47, 50, 52, 55, 59, 64, 65, 67, 72, 73, 74, 77, 78, 80, 83, 84, 92, 93, 95, 96, 98]
    # bad_dataset, good_dataset = get_voting_training_dataloader_by_class(bad_index=bad_set_index, mean=settings.CIFAR100_TRAIN_MEAN, std=settings.CIFAR100_TRAIN_STD, batch_size=args.b, num_workers=2, shuffle=True, dataset_name=args.dataset_name, train=True)
    # bad_test_loader, good_test_loader = get_voting_training_dataloader_by_class(bad_index=bad_set_index, mean=settings.CIFAR100_TRAIN_MEAN, std=settings.CIFAR100_TRAIN_STD, batch_size=args.b, num_workers=2, shuffle=True, dataset_name=args.dataset_name, train=False)
    #
    # train_model(model_args=args, train_loader=bad_dataset, test_loader=bad_test_loader)
    # train_model(model_args=args, train_loader=good_dataset, test_loader=good_test_loader)

    # model = get_network(args)
    # model.load_state_dict(torch.load('checkpoint/resnet18/resnet18-23-best_test0.7458best_train0.99238.pth', map_location=settings.CUDA))
    # loss_function = nn.CrossEntropyLoss()
    # training_acc, training_loss = test_model(model_args=args, model=model, test_loader=training_loader, loss_function=loss_function)
    # print(training_acc, training_loss)
    #
    # test_acc, test_loss = test_model(model_args=args, model=model, test_loader=testing_loader, loss_function=loss_function)
    # print(test_acc, test_loss)
