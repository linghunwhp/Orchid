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
from utils import get_network, get_retraining_one_class_dataloader, get_test_dataloader_all, get_retraining_dataloader


def train(model_args, model, train_loader, optimizer):
    model.train()
    for images, labels in train_loader:
        if model_args.gpu:
            images = images.to(settings.CUDA)
            labels = labels.to(settings.CUDA)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        del images, labels
        if model_args.gpu:
            with torch.cuda.device(settings.CUDA):
                torch.cuda.empty_cache()


@torch.no_grad()
def test(model_args, model, test_loader):
    model.eval()

    test_loss = 0.0  # cost function error
    correct = 0.0

    for images, labels in test_loader:
        if model_args.gpu:
            images = images.to(settings.CUDA)
            labels = labels.to(settings.CUDA)

        outputs = model(images)
        loss = loss_function(outputs, labels)
        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

        del images, labels
        if model_args.gpu:
            with torch.cuda.device(settings.CUDA):
                torch.cuda.empty_cache()

    return float(correct) / len(test_loader.dataset), test_loss


def train_model(model_args, train_loader, valid_loader):
    attack_x = None
    attack_y = None
    attack_classes = None

    target_model = get_network(model_args)
    target_model_optimizer = optim.SGD(target_model.parameters(), lr=model_args.lr, momentum=0.9, weight_decay=1e-4)
    target_model_scheduler = optim.lr_scheduler.MultiStepLR(target_model_optimizer, milestones=settings.MILESTONES, gamma=0.2)    # learning rate decay

    best_acc = 0.0
    best_acc_epoch = 0
    best_model_state_dict = None
    for epoch in range(1, settings.EPOCH):
        start = time.time()

        if epoch > model_args.warm:
            target_model_scheduler.step()
        train(model_args=model_args, model=target_model, train_loader=train_loader, optimizer=target_model_optimizer)    # train target model
        train_acc, train_loss = test(model_args=model_args, model=target_model, test_loader=train_loader)
        valid_acc, valid_loss = test(model_args=model_args, model=target_model, test_loader=valid_loader)

        # start to save best performance model after learning rate decay to 0.01
        if valid_acc - best_acc > 0.001:
            best_acc = valid_acc
            best_acc_epoch = epoch
            best_model_state_dict = deepcopy(target_model.state_dict())

        if (epoch - best_acc_epoch) >= 10:
            torch.save(best_model_state_dict, checkpoint_path.format(net=model_args.net, epoch=best_acc_epoch, type='best_retrained_model_'+str(round(best_acc, 4))))
            target_model.load_state_dict(best_model_state_dict)
            break

        finish = time.time()
        # print(f'Target Model: Epoch: {epoch}, Learning rate: {target_model_optimizer.param_groups[0]["lr"]:.4f}, Valid loss: {valid_loss :.4f}, Valid acc: {valid_acc:.4f}, Time consumed:{finish - start:.2f}s')
        print(f'Target Model: Epoch: {epoch}, Learning rate: {target_model_optimizer.param_groups[0]["lr"]:.4f}, Train loss: {train_loss :.4f}, Train acc: {train_acc:.4f}, Valid loss: {valid_loss :.4f}, Valid acc: {valid_acc:.4f}, Time consumed:{finish - start:.2f}s')

    return attack_x, attack_y, attack_classes


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-b', type=int, default=128, help='batch size for data loader')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument('-del_ratio', type=float, default=0.1, help='left data ratio')
    parser.add_argument('-num_class', type=float, default=100, help='class number')
    parser.add_argument('-dataset_name', type=str, default='cifar100', help='dataset name')
    args = parser.parse_args()
    net = get_network(args)

    original_training_dataset_loader = torch.load("checkpoint/allcnn/Sunday_27_December_2020_22h_27m_37s/allcnn-0-cifar100_target_train_loader.pth")

    training_left_loader, training_deleted_loader = get_retraining_one_class_dataloader(
        dataset_loader=original_training_dataset_loader,
        delete_class=50,
        batch_size=args.b,
        num_workers=4,
        shuffle=True
    )

    # training_left_loader, training_deleted_loader = get_retraining_dataloader(
    #     dataset_loader=original_training_dataset_loader,
    #     delete_num=100,
    #     batch_size=args.b,
    #     num_workers=4,
    #     shuffle=True
    # )

    validation_loader = get_test_dataloader_all(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        batch_size=args.b,
        shuffle=True,
        dataset_name=args.dataset_name
    )

    # print(len(training_left_loader.dataset))
    # print(len(training_deleted_loader.dataset))
    # print([b for a,b in training_deleted_loader])

    loss_function = nn.CrossEntropyLoss()
    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)

    # create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    torch.save(training_left_loader, checkpoint_path.format(net=args.net, epoch=0, type=args.dataset_name+'_training_left_loader'))
    torch.save(training_deleted_loader, checkpoint_path.format(net=args.net, epoch=0, type=args.dataset_name+'_training_deleted_loader'))

    main_process_start = time.time()

    train_model(args, training_left_loader, validation_loader)
    # train_model(args, original_training_dataset_loader, validation_loader)

    main_process_finish = time.time()

    print(f'Full training time consumed: {main_process_finish - main_process_start:.2f}s')


