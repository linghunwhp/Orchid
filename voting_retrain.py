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
from utils import get_network, get_test_dataloader_all, get_voting_training_dataloader_by_class, get_voting_training_dataloader, train_one_epoch, voting, test_model
from matplotlib import pyplot as plt


def train_model(model_args, train_loader, test_loader, model_index):
    model = get_network(model_args)
    model_optimizer = optim.SGD(model.parameters(), lr=model_args.lr, momentum=0.9, weight_decay=1e-4)
    model_scheduler = optim.lr_scheduler.MultiStepLR(model_optimizer, milestones=settings.MILESTONES, gamma=0.1)    # learning rate decay
    loss_function = nn.CrossEntropyLoss()

    best_test_acc = 0.0
    best_train_acc = 0.0
    best_epoch = 0
    best_model_state_dict = None

    train_acc_list = []
    train_loss_list = []
    test_acc_list = []
    test_loss_list = []

    for epoch in range(1, settings.EPOCH):
        start = time.time()

        if epoch > model_args.warm:
            model_scheduler.step()
        train_acc, train_loss = train_one_epoch(model_args=model_args, model=model, train_loader=train_loader, optimizer=model_optimizer, loss_function=loss_function)    # train model
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

        finish = time.time()
        print(f'Model {model_index}: Epoch: {epoch} Learning rate: {model_optimizer.param_groups[0]["lr"]:.4f} Train loss: {train_loss :.4f} Train acc: {train_acc:.4f} Valid loss: {test_loss :.4f} Valid acc: {test_acc:.4f} Time consumed:{finish - start:.2f}s')

    plt.title('Train and Test Accuracy')
    plt.plot(range(1, len(train_acc_list) + 1), train_acc_list, label='Train Accuracy')
    plt.plot(range(1, len(test_acc_list) + 1), test_acc_list, label='Test Accuracy')
    plt.legend()
    plt.savefig(checkpoint_path.format(net=model_args.net, epoch="", type="Accuracy") + '_' + str(model_index) + "_.png")
    plt.close()

    plt.title('Train and Test Loss')
    plt.plot(range(1, len(train_loss_list) + 1), train_loss_list, label='Train Loss')
    plt.plot(range(1, len(test_loss_list) + 1), test_loss_list, label='Test Loss')
    plt.legend()
    plt.savefig(checkpoint_path.format(net=model_args.net, epoch="", type="Loss") + '_' + str(model_index) + "_.png")
    plt.close()
    model.load_state_dict(best_model_state_dict)
    torch.save(best_model_state_dict, checkpoint_path.format(net=model_args.net, epoch=best_epoch, type='test' + str(round(best_test_acc, 4)) + 'train' + str(round(best_train_acc, 4)) + '_' + str(model_index)))
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-b', type=int, default=64, help='batch size for data loader')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('-del_ratio', type=float, default=0.1, help='left data ratio')
    parser.add_argument('-num_class', type=float, default=100, help='class number')
    parser.add_argument('-dataset_name', type=str, default='cifar100', help='dataset name')
    args = parser.parse_args()

    deleted_loader, left_loader = get_voting_training_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        equal_divide_n=5,
        batch_size=args.b,
        num_workers=4,
        shuffle=True,
        dataset_name=args.dataset_name
    )

    testing_loader = get_test_dataloader_all(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        batch_size=args.b,
        shuffle=True,
        dataset_name=args.dataset_name
    )

    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    model_list = []
    for index in range(len(left_loader)):
        torch.save(left_loader[index], checkpoint_path.format(net=args.net, epoch=0, type=args.dataset_name+'_training_left_loader_'+str(index)))
        torch.save(deleted_loader[index], checkpoint_path.format(net=args.net, epoch=0, type=args.dataset_name+'_training_deleted_loader_'+str(index)))

        main_process_start = time.time()
        model_list.append(train_model(model_args=args, train_loader=left_loader[index], test_loader=testing_loader, model_index=index))
        main_process_finish = time.time()
        print(f'Full training time consumed: {main_process_finish - main_process_start:.2f}s')

    voting_accuracy, all_unreached, first_two_unreached = voting(args, model_list, testing_loader)
    print(f"Voting accuracy {voting_accuracy} all_unreached: {all_unreached} first_two_unreached: {first_two_unreached}")
