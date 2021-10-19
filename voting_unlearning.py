from copy import deepcopy
import time
import os
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from conf import settings
from utils import get_network, get_test_dataloader_all, test_model, voting, eval_uniform_distribution, UniformNormLossFunc, train_one_epoch, adjustment_step, partition_training_data_loader, test_each_class
torch.set_printoptions(profile="full")


def fixing(model_args, model, training_left_loader, train_valid_loader, loss_function, optim_fix, epoch_n):
    best_fixing_test_acc = 0.0
    best_fixing_epoch = 0
    best_fixing_model = None
    best_avg_fixing_loss = 0.0
    fixing_loss = 0.0

    start = time.time()
    for ep in range(1, 51):
        _, fixing_loss_temp = train_one_epoch(model_args=model_args, model=model, train_loader=training_left_loader, optimizer=optim_fix, loss_function=loss_function)
        fixing_loss += fixing_loss_temp
        fixing_test_acc, _ = test_model(model_args=model_args, model=model, test_loader=train_valid_loader, loss_function=loss_function)
        print(f"fixing_test_acc is: {fixing_test_acc}, epoch is: {ep}")
        if fixing_test_acc - best_fixing_test_acc > 0.001:
            best_fixing_test_acc = fixing_test_acc
            best_fixing_epoch = ep
            best_avg_fixing_loss = fixing_loss/ep
            best_fixing_model = deepcopy(model.state_dict())
        elif (ep - best_fixing_epoch) >= 5:
            model.load_state_dict(best_fixing_model)
            break

    finish = time.time()
    print('epoch {} training time consumed: {:.2f}s'.format(epoch_n, finish - start))
    return best_avg_fixing_loss


def unlearn_original_model(model_args, deleted_loader_list, left_loader_list, test_loader, loss_function_origin, loss_function_update, blunt_strategy='uniform'):
    # for distribution blunt
    test_acc_distribution_list = []
    if blunt_strategy == 'distribution':
        # cifar100 - mobilenet
        # test_acc_distribution_list = [0.9000, 0.8200, 0.5000, 0.3800, 0.4200, 0.6700, 0.7500, 0.6600, 0.8000, 0.6800,
        #                               0.4500, 0.3500, 0.7500, 0.6300, 0.6400, 0.6500, 0.7000, 0.8100, 0.6200, 0.5600,
        #                               0.8700, 0.8100, 0.5500, 0.8000, 0.8000, 0.5000, 0.5900, 0.5500, 0.7000, 0.6200,
        #                               0.5200, 0.6500, 0.5800, 0.6000, 0.6700, 0.4400, 0.7000, 0.7300, 0.5200, 0.8000,
        #                               0.5900, 0.8200, 0.7100, 0.7300, 0.3600, 0.5500, 0.4000, 0.6300, 0.9200, 0.8600,
        #                               0.4300, 0.6400, 0.6600, 0.9000, 0.7900, 0.3100, 0.8800, 0.7400, 0.7500, 0.6900,
        #                               0.8700, 0.7200, 0.7000, 0.6300, 0.4500, 0.4500, 0.7100, 0.5700, 0.9100, 0.7700,
        #                               0.6300, 0.7200, 0.4000, 0.5300, 0.5100, 0.8600, 0.8900, 0.5000, 0.5800, 0.7400,
        #                               0.4500, 0.6400, 0.8800, 0.5400, 0.6100, 0.7700, 0.6300, 0.7300, 0.7600, 0.7900,
        #                               0.7400, 0.7700, 0.5600, 0.3700, 0.8800, 0.7100, 0.5300, 0.6600, 0.3500, 0.6500]

        # cifar100 - shufflenetv2
        test_acc_distribution_list = [0.8600, 0.7800, 0.5300, 0.4400, 0.6000, 0.7800, 0.7000, 0.7100, 0.8600, 0.7700,
                                      0.4900, 0.4400, 0.7900, 0.4700, 0.6300, 0.7400, 0.7400, 0.7700, 0.6600, 0.6100,
                                      0.8700, 0.7900, 0.6600, 0.8300, 0.8700, 0.5600, 0.5800, 0.5700, 0.8200, 0.6700,
                                      0.6100, 0.6800, 0.6400, 0.6200, 0.7200, 0.4400, 0.8200, 0.7200, 0.6000, 0.7900,
                                      0.6200, 0.8600, 0.6900, 0.7600, 0.5400, 0.5800, 0.5200, 0.6700, 0.8500, 0.8200,
                                      0.5700, 0.6800, 0.5300, 0.8800, 0.7300, 0.4700, 0.8300, 0.7400, 0.7700, 0.6200,
                                      0.8800, 0.7500, 0.7500, 0.7300, 0.4900, 0.5400, 0.7500, 0.5800, 0.9500, 0.7800,
                                      0.7600, 0.7600, 0.4100, 0.5300, 0.6400, 0.8300, 0.8700, 0.6500, 0.6200, 0.7600,
                                      0.5200, 0.7200, 0.8800, 0.5300, 0.6400, 0.8100, 0.6000, 0.7500, 0.7500, 0.8000,
                                      0.8000, 0.7900, 0.5900, 0.4600, 0.9600, 0.6500, 0.6800, 0.6700, 0.5200, 0.7500]
        # model_original = get_network(model_args)
        # model_original.load_state_dict(torch.load(model_args.weights, map_location=settings.CUDA))
        # test_acc_tensor_by_class = torch.tensor(test_each_class(model_args=model_args, model=model_original, loss_function=loss_function_origin, train=False))
        # print(test_acc_tensor_by_class)

    if blunt_strategy == 'random':
        test_acc_distribution_list = np.random.rand(model_args.num_class)

    sub_model_list = []
    for i in range(model_args.num_sub_models):
        torch.save(left_loader_list[i], checkpoint_path.format(net=model_args.net, epoch=0, type=model_args.dataset_name+'_training_left_loader_'+str(i)))
        torch.save(deleted_loader_list[i], checkpoint_path.format(net=model_args.net, epoch=0, type=model_args.dataset_name+'_training_deleted_loader_'+str(i)))

        model_original = get_network(model_args)
        model_original.load_state_dict(torch.load(model_args.weights, map_location=settings.CUDA))

        if i == model_args.num_sub_models:
            sub_model_list.append(model_original)
            break

        optimizer_adjust = optim.SGD(model_original.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
        # optimizer_adjust = optim.SGD(model_original.parameters(), lr=0.002, momentum=0.9, weight_decay=1e-4)
        optimizer_fix = optim.SGD(model_original.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

        best_epoch = 0
        best_model_state_dict = None
        all_epochs = 16
        weighted = 0.01   # hyper parameter to balance the CE loss and regularization part
        kl_loss_list = []
        ce_loss_list = []
        total_loss_list = []
        fixed_model_test_acc_list = []
        fixed_model_deleted_data_acc_list = []

        original_test_acc, original_test_loss = test_model(model_args=model_args, model=model_original, test_loader=test_loader, loss_function=loss_function_origin)
        fixed_model_test_acc_list.append(original_test_acc)

        best_kl_loss = eval_uniform_distribution(model_args=model_args, model=model_original, test_loader=deleted_loader_list[i], n_classes=model_args.num_class, loss_function=loss_function_update)
        kl_loss_list.append(weighted * best_kl_loss)

        _, original_left_loss = test_model(model_args=model_args, model=model_original, test_loader=left_loader_list[i], loss_function=loss_function_origin)
        ce_loss_list.append(original_left_loss)

        best_total_loss = weighted * best_kl_loss + original_left_loss
        total_loss_list.append(best_total_loss)

        original_model_deleted_data_acc, _ = test_model(model_args=model_args, model=model_original, test_loader=deleted_loader_list[i], loss_function=loss_function_origin)
        fixed_model_deleted_data_acc_list.append(original_model_deleted_data_acc)

        for epoch in range(1, all_epochs):
            adjustment_step(model_args=model_args, model=model_original, dataset_loader=deleted_loader_list[i], loss_function=loss_function_update, optimizer=optimizer_adjust, blunt_strategy=blunt_strategy, test_acc_distribution_list=test_acc_distribution_list)
            fixing(model_args=model_args, model=model_original, training_left_loader=left_loader_list[i], train_valid_loader=test_loader, loss_function=loss_function_origin, optim_fix=optimizer_fix, epoch_n=epoch)

            _, ce_loss = test_model(model_args=model_args, model=model_original, test_loader=left_loader_list[i], loss_function=loss_function_origin)
            ce_loss_list.append(ce_loss)

            kl_deleted_loss = eval_uniform_distribution(model_args=model_args, model=model_original, test_loader=deleted_loader_list[i], n_classes=model_args.num_class, loss_function=loss_function_update)
            kl_loss_list.append(weighted * kl_deleted_loss)

            total_loss = weighted * kl_deleted_loss + ce_loss
            total_loss_list.append(total_loss)

            fixed_model_test_acc, _ = test_model(model_args=model_args, model=model_original, test_loader=test_loader, loss_function=loss_function_origin)
            fixed_model_test_acc_list.append(fixed_model_test_acc)

            fixed_model_deleted_data_acc, _ = test_model(model_args=model_args, model=model_original, test_loader=deleted_loader_list[i], loss_function=loss_function_origin)
            fixed_model_deleted_data_acc_list.append(fixed_model_deleted_data_acc)

            print(f'Fixing Epoch: {epoch} Total_loss: {total_loss}, KL_loss: {kl_deleted_loss}, CE_loss: {ce_loss} '
                  f'fixed model test accuracy: {fixed_model_test_acc:0.4} kld testing loss: {weighted * kl_deleted_loss:0.4f} '
                  f'fixed model deleted data accuracy: {fixed_model_deleted_data_acc:0.4f}')

            # save best performance model by the kl-d similarity, the lower the better
            # if kl_deleted_loss < best_kl_loss:
            if epoch == (all_epochs-1):
                best_kl_loss = kl_deleted_loss
                best_epoch = epoch
                best_model_state_dict = deepcopy(model_original.state_dict())

            if epoch%5 == 0:
                plt.title('Fixed model test accuracy on test and deleted data')
                plt.plot(range(0, len(fixed_model_test_acc_list)), fixed_model_test_acc_list, label='test accuracy on test data')
                plt.plot(range(0, len(fixed_model_deleted_data_acc_list)), fixed_model_deleted_data_acc_list, label='test accuracy on deleted data')
                plt.legend()
                plt.savefig(checkpoint_path.format(net=model_args.net, epoch=epoch, type="Accuracy_") + str(i) + "_.png")
                plt.close()

                plt.title('The total fixing loss')
                plt.plot(range(0, len(total_loss_list)), total_loss_list, label='Total loss')
                plt.legend()
                plt.savefig(checkpoint_path.format(net=model_args.net, epoch=epoch, type="Total_loss_") + str(i) + "_.png")
                plt.close()

                plt.title('The kl loss on deleted data and ce loss on left data')
                plt.plot(range(0, len(kl_loss_list)), kl_loss_list, label='kl loss')
                plt.plot(range(0, len(ce_loss_list)), ce_loss_list, label='ce loss')
                plt.legend()
                plt.savefig(checkpoint_path.format(net=model_args.net, epoch=epoch, type="average_fixing_testing_kl_loss_") + str(i) + "_.png")
                plt.close()

        model_original.load_state_dict(best_model_state_dict)
        train_acc, _ = test_model(model_args=model_args, model=model_original, test_loader=left_loader_list[i], loss_function=loss_function_origin)
        test_acc, _ = test_model(model_args=model_args, model=model_original, test_loader=test_loader, loss_function=loss_function_origin)
        torch.save(best_model_state_dict, checkpoint_path.format(net=model_args.net, epoch=best_epoch, type='best_test' + str(round(test_acc, 4)) + 'best_train' + str(round(train_acc, 4))+'_'+str(i)))
        sub_model_list.append(model_original)

    return sub_model_list


def updating(model_args, model, train_loader, output_vector, optimizer, loss_function, loss_function_update):
    model.train()
    train_loss_temp = 0.0
    correct = 0
    current_length = 0

    for _, (images, labels) in enumerate(train_loader):
        if model_args.gpu:
            images = images.to(settings.CUDA)
            labels = labels.to(settings.CUDA)

        optimizer.zero_grad()
        outputs = model(images)

        output_vector_temp = output_vector[current_length:current_length + len(images)].to(settings.CUDA)
        current_length += len(images)

        # just with l2-norm loss, result is good
        # loss = loss_function_update(outputs, output_vector_temp)
        # loss = 100*loss_function(outputs, labels) + loss_function_update(outputs, output_vector_temp)
        loss = 150*loss_function(outputs, labels) + loss_function_update(outputs, output_vector_temp)

        train_loss_temp += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()
        loss.backward()
        optimizer.step()

        del images, labels
        if model_args.gpu:
            with torch.cuda.device(settings.CUDA):
                torch.cuda.empty_cache()

    return float(correct)/len(train_loader.dataset), train_loss_temp


def get_confidence_predict(model_args, model, train_loader):
    one_model_outputs = None
    one_model_predict = None

    model.eval()
    with torch.no_grad():
        for images, labels in train_loader:
            if model_args.gpu:
                images = images.to(settings.CUDA)
                labels = labels.to(settings.CUDA)

            outputs = model(images)
            if one_model_outputs is None:
                one_model_outputs = outputs
            else:
                one_model_outputs = torch.cat((one_model_outputs, outputs))

            _, predicts = outputs.max(1)
            if one_model_predict is None:
                one_model_predict = predicts.eq(labels)
            else:
                one_model_predict = torch.cat((one_model_predict, predicts.eq(labels)))

            # if one_model_predict is None:
            #     one_model_predict = predicts
            # else:
            #     one_model_predict = torch.cat((one_model_predict, predicts))

            del images, labels
            if model_args.gpu:
                with torch.cuda.device(settings.CUDA):
                    torch.cuda.empty_cache()

    return one_model_outputs, one_model_predict


# calculate the prediction result variance of E among models in GSS
def variance_GSS(model_args, sub_model_list, test_loader):
    sub_model_train_predict = []

    for i in range(model_args.num_sub_models):
        _, one_model_predict = get_confidence_predict(model_args=model_args, model=sub_model_list[i], train_loader=test_loader)
        sub_model_train_predict.append(one_model_predict.tolist())

    model_original = get_network(model_args)
    model_original.load_state_dict(torch.load(model_args.weights, map_location=settings.CUDA))
    _, original_model_test_predicts = get_confidence_predict(model_args=model_args, model=model_original, train_loader=test_loader)

    for i in range(model_args.num_sub_models):
        # print(sum(np.array(original_model_test_predicts.tolist()) == np.array(sub_model_train_predict[i])))
        print(np.var(np.array(original_model_test_predicts.tolist()), np.array(sub_model_train_predict[i])))


# average all the correct vector
def combine_confidence_vector_origin(model_args, sub_model_list, train_loader):
    sub_model_train_outputs = []
    sub_model_train_predict = []

    for i in range(model_args.num_sub_models):
        one_model_outputs, one_model_predict = get_confidence_predict(model_args=model_args, model=sub_model_list[i], train_loader=train_loader)
        sub_model_train_outputs.append(one_model_outputs.tolist())
        sub_model_train_predict.append(one_model_predict.tolist())

    sub_model_train_outputs = torch.tensor(sub_model_train_outputs)
    sub_model_train_predict = torch.tensor(sub_model_train_predict)

    model_original = get_network(model_args)
    model_original.load_state_dict(torch.load(model_args.weights, map_location=settings.CUDA))
    original_model_train_outputs, _ = get_confidence_predict(model_args=model_args, model=model_original, train_loader=train_loader)

    smallest_correct_output = []
    avg_correct_output = []
    largest_correct_output = []
    for i in range(len(train_loader.dataset)):
        correct_index = torch.where(sub_model_train_predict[:, i] == True)[0]
        sum_output = None

        if len(correct_index) > 0:
            smallest_index = correct_index[0]
            largest_index = correct_index[0]
            for j in correct_index:
                if sub_model_train_outputs[j, i].max() < sub_model_train_outputs[smallest_index, i].max():
                    smallest_index = j

                if sub_model_train_outputs[j, i].max() > sub_model_train_outputs[largest_index, i].max():
                    largest_index = j

                if sum_output is None:
                    sum_output = sub_model_train_outputs[j, i]
                else:
                    sum_output += sub_model_train_outputs[j, i]

            smallest_correct_output.append(sub_model_train_outputs[smallest_index, i].tolist())
            avg_correct_output.append((sum_output/len(correct_index)).tolist())
            largest_correct_output.append(sub_model_train_outputs[largest_index, i].tolist())
        else:
            smallest_correct_output.append(original_model_train_outputs[i].tolist())
            avg_correct_output.append(original_model_train_outputs[i].tolist())
            largest_correct_output.append(original_model_train_outputs[i].tolist())

    return smallest_correct_output, avg_correct_output, largest_correct_output


def update_origianl_model(model_args, sub_model_list, train_loader, test_loader, loss_function, loss_function_update):
    smallest_correct_output, avg_correct_output, largest_correct_output = combine_confidence_vector_origin(model_args=model_args, sub_model_list=sub_model_list, train_loader=train_loader)

    # fix original model by average good vector
    model_original = get_network(model_args)
    model_original.load_state_dict(torch.load(model_args.weights, map_location=settings.CUDA))
    optimizer = optim.SGD(model_original.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 30, 45], gamma=0.1)

    train_acc_list = []
    train_loss_list = []
    test_acc_list = []
    test_loss_list = []

    best_test_acc, test_loss = test_model(model_args=model_args, model=model_original, test_loader=test_loader, loss_function=loss_function)
    print("best_test_acc is ", best_test_acc)
    test_acc_list.append(best_test_acc)
    test_loss_list.append(test_loss)

    for epoch in range(1, 61):
        if epoch > model_args.warm:
            train_scheduler.step()

        train_acc, train_loss = updating(model_args=model_args, model=model_original, train_loader=train_loader, output_vector=torch.tensor(avg_correct_output), optimizer=optimizer, loss_function=loss_function, loss_function_update=loss_function_update)
        train_acc_list.append(train_acc)
        train_loss_list.append(train_loss)

        test_acc, test_loss = test_model(model_args=model_args, model=model_original, test_loader=test_loader, loss_function=loss_function)
        test_acc_list.append(test_acc)
        test_loss_list.append(test_loss)

        if test_acc - best_test_acc > 0.0001:
            best_train_acc = train_acc
            best_test_acc = test_acc
            best_epoch = epoch
            torch.save(model_original.state_dict(), checkpoint_path.format(net=model_args.net, epoch=best_epoch, type='best_test' + str(best_test_acc) + 'best_train' + str(best_train_acc)))

        print(f'Epoch: {epoch} Learning rate: {optimizer.param_groups[0]["lr"]:.5f} Train Loss: {train_loss:.4f} Test Loss: {test_loss:.4f} Train Acc: {train_acc:.4f} Test Acc: {test_acc:.4f}')

        if epoch % 10 == 0:
            plt.title('Train and Test Accuracy')
            plt.plot(range(len(train_acc_list)), train_acc_list, label='Train Accuracy')
            plt.plot(range(len(test_acc_list)), test_acc_list, label='Test Accuracy')
            plt.legend()
            plt.savefig(checkpoint_path.format(net=model_args.net, epoch=epoch, type="Accuracy_") + str(epoch) + "_.png")
            plt.close()

            plt.title('Train and Test Loss')
            plt.plot(range(len(train_loss_list)), train_loss_list, label='Train Loss')
            plt.plot(range(len(test_loss_list)), test_loss_list, label='Test Loss')
            plt.legend()
            plt.savefig(checkpoint_path.format(net=model_args.net, epoch=epoch, type="Loss_") + str(epoch) + "_.png")
            plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-weights', type=str, required=True, help='the weights file you want to test')
    parser.add_argument('-weights2', type=str, required=False, help='the weights file you want to test')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-b', type=int, default=128, help='batch size for data loader')
    parser.add_argument('-lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument('-ratio', type=float, default=0.9, help='left data ratio')
    parser.add_argument('-num_class', type=float, default=100, help='class number')
    parser.add_argument('-dataset_name', type=str, default='cifar100', help='dataset name')
    parser.add_argument('-num_sub_models', type=int, default=8, help='number of sub models')

    args = parser.parse_args()
    loss_function_ce = nn.CrossEntropyLoss()
    loss_function_norm = UniformNormLossFunc()

    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    testing_loader = get_test_dataloader_all(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        batch_size=args.b,
        shuffle=True,
        dataset_name=args.dataset_name
    )

    training_loader, blunt_data_loader, left_data_loader, _ = partition_training_data_loader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_sub_models=args.num_sub_models,
        batch_size=args.b,
        num_workers=4,
        shuffle=True,
        dataset_name=args.dataset_name
    )

    # unlearn_model_list = unlearn_original_model(
    #     model_args=args,
    #     deleted_loader_list=blunt_data_loader,
    #     left_loader_list=left_data_loader,
    #     test_loader=testing_loader,
    #     loss_function_origin=loss_function_ce,
    #     loss_function_update=loss_function_norm,
    #     blunt_strategy='random'
    #     # blunt_strategy='distribution'
    # )
    # voting_accuracy = voting(args, unlearn_model_list, testing_loader)
    # print(f"Voting accuracy is {voting_accuracy}")

    unlearn_model_list = []
    for index in range(args.num_sub_models):
        original_model = get_network(args)
        # original_model.load_state_dict(torch.load('checkpoint/CIFAR100/mobilenet/Unlearned_all/' + str(args.num_sub_models) + '_' + str(index) + '.pth', map_location=settings.CUDA))
        original_model.load_state_dict(torch.load('checkpoint/CIFAR100/mobilenet/old_8/' + str(args.num_sub_models) + '_' + str(index) + '.pth', map_location=settings.CUDA))
        train_acc, _ = test_model(model_args=args, model=original_model, test_loader=training_loader, loss_function=loss_function_ce)
        test_acc, _ = test_model(model_args=args, model=original_model, test_loader=testing_loader, loss_function=loss_function_ce)
        print(f"Train Acc is {train_acc}, Test Acc is {test_acc}")
        unlearn_model_list.append(original_model)

    # variance_GSS(model_args=args, sub_model_list=unlearn_model_list, test_loader=testing_loader)

    # voting_accuracy = voting(args, unlearn_model_list, testing_loader)
    # print(f"Voting accuracy is {voting_accuracy}")

    update_origianl_model(model_args=args, sub_model_list=unlearn_model_list, train_loader=training_loader, test_loader=testing_loader, loss_function=loss_function_ce, loss_function_update=loss_function_norm)
