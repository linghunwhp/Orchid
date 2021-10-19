# test.py
# !/usr/bin/env python3

import argparse
import torch
from conf import settings
from utils import get_network, get_training_dataloader, get_test_dataloader, get_one_class_other_dataloader, get_test_dataloader_all, test_model


def eval_uniform_distribution(model, test_loader):
    model.eval()
    sf = torch.nn.LogSoftmax(dim=0)
    loss_function_kld = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
    kl_d = 0.0
    with torch.no_grad():
        for n_iter, (image, label) in enumerate(test_loader):
            if args.gpu:
                image = image.cuda()

            output = sf(model(image))
            uniform_tensor = sf(torch.full([len(image), n_classes], 1/n_classes)).cuda()
            kl_d += loss_function_kld(output, uniform_tensor)

        average_kl_loss = kl_d / len(test_loader)
        print("kl_distribution_similarity is: ", average_kl_loss)
        return average_kl_loss


@torch.no_grad()
def test(model, test_loader):
    model.eval()
    train_correct_1 = 0.0
    train_correct_5 = 0.0

    with torch.no_grad():
        for n_iter, (image, label) in enumerate(test_loader):
            # print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(cifar100_training_loader)))

            if args.gpu:
                image = image.cuda()
                label = label.cuda()

            output = model(image)
            _, pred = output.topk(5, 1, largest=True, sorted=True)

            label = label.view(label.size(0), -1).expand_as(pred)
            correct = pred.eq(label).float()

            # compute top1
            train_correct_1 += correct[:, :1].sum()

            # compute top 5
            train_correct_5 += correct[:, :5].sum()

    # print("Top 1 acc: ", train_correct_1 / len(test_loader.dataset))
    # print("Top 5 acc: ", train_correct_5 / len(test_loader.dataset))
    # print("Parameter numbers: {}".format(sum(p.numel() for p in model.parameters())))
    return float(train_correct_1)/len(test_loader.dataset), float(train_correct_5)/len(test_loader.dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-weights', type=str, required=True, help='the weights file you want to test')
    parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('-b', type=int, default=128, help='batch size for data loader')
    parser.add_argument('-num_class', type=float, default=100, help='class number')
    parser.add_argument('-dataset_name', type=str, default='cifar100', help='dataset name')
    args = parser.parse_args()

    net = get_network(args)
    net.load_state_dict(torch.load(args.weights))

    file_1 = open("structure.txt", "w")
    file_1.write(str(net))
    file_1.close()
    file_2 = open("weight_matrix.txt", "w")
    file_2.write(str(torch.load(args.weights)))
    file_2.close()

    n_classes = args.num_class

    # # training accuracy on train data
    testing_loader = get_test_dataloader_all(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        batch_size=args.b,
        shuffle=True,
        dataset_name=args.dataset_name
    )

    test_acc, _ = test(net, testing_loader)
    print("test_acc is ", test_acc)

    # # test accuracy on test data
    # validation_loader, testing_loader = get_test_dataloader(
    #     settings.CIFAR10_TRAIN_MEAN,
    #     settings.CIFAR10_TRAIN_STD,
    #     num_workers=4,
    #     batch_size=args.b,
    #     dataset_name=args.dataset_name
    # )

    # for mobilenet cifar100
    # training_left_loader = torch.load("checkpoint/mobilenet/Monday_09_November_2020_13h_50m_45s/mobilenet-0-cifar100_training_left_loader.pth")
    # training_deleted_loader = torch.load("checkpoint/mobilenet/Monday_09_November_2020_13h_50m_45s/mobilenet-0-cifar100_training_deleted_loader.pth")

    # for mobilenet cifar10
    # training_left_loader = torch.load("checkpoint/mobilenet/Saturday_14_November_2020_15h_06m_44s/mobilenet-0-cifar10_training_left_loader.pth")
    # training_deleted_loader = torch.load("checkpoint/mobilenet/Saturday_14_November_2020_15h_06m_44s/mobilenet-0-cifar10_training_deleted_loader.pth")

    # for allcnn cifar100
    #  python unlearning.py -net allcnn -weights checkpoint/allcnn/Saturday_14_November_2020_16h_20m_43s/allcnn-124-best.pth -weights2 checkpoint/allcnn/Saturday_14_November_2020_18h_05m_29s/allcnn-129-best.pth -gpu
    # training_left_loader = torch.load("checkpoint/allcnn/Saturday_14_November_2020_18h_05m_29s/allcnn-0-cifar100_training_left_loader.pth")
    # training_deleted_loader = torch.load("checkpoint/allcnn/Saturday_14_November_2020_18h_05m_29s/allcnn-0-cifar100_training_deleted_loader.pth")

    # for allcnn cifar10
    # python unlearning.py -net allcnn -weights checkpoint/allcnn/Friday_13_November_2020_22h_45m_04s/allcnn-171-best.pth -weights2 checkpoint/allcnn/Friday_13_November_2020_22h_45m_39s/allcnn-129-best.pth -gpu
    # training_left_loader = torch.load("checkpoint/allcnn/Friday_13_November_2020_22h_45m_39s/allcnn-0-cifar10_training_left_loader.pth")
    # training_deleted_loader = torch.load("checkpoint/allcnn/Friday_13_November_2020_22h_45m_39s/allcnn-0-cifar10_training_deleted_loader.pth")
    #
    # # training accuracy
    # test(net, training_left_loader)
    #
    # test(net, training_deleted_loader)
    #
    # # test accuracy
    # test(net, testing_loader)
    #
    # # similarity between deleted dataset and uniform distribution
    # eval_uniform_distribution(net, training_left_loader)
    # eval_uniform_distribution(net, training_deleted_loader)
    # eval_uniform_distribution(net, testing_loader)

    # training_acc_list_all_class = []
    # testing_acc_list_all_class = []
    #
    # for i in range(args.num_class):
    #     train_loader, _ = get_one_class_other_dataloader(
    #         settings.CIFAR10_TRAIN_MEAN,
    #         settings.CIFAR10_TRAIN_STD,
    #         train_class=i,
    #         train=True,
    #         batch_size=args.b,
    #         num_workers=4,
    #         shuffle=True,
    #         dataset_name=args.dataset_name
    #     )
    #
    #     valid_loader, _ = get_one_class_other_dataloader(
    #         settings.CIFAR10_TRAIN_MEAN,
    #         settings.CIFAR10_TRAIN_STD,
    #         train_class=i,
    #         train=False,
    #         batch_size=args.b,
    #         num_workers=4,
    #         shuffle=True,
    #         dataset_name=args.dataset_name
    #     )
    #
    #     training_acc, _ = test(net, train_loader)
    #     training_acc_list_all_class.append(round(training_acc, 4))
    #
    #     testing_acc, _ = test(net, valid_loader)
    #     testing_acc_list_all_class.append(round(testing_acc, 4))
    #
    #     print(f'Class:{i} training_acc:{training_acc} testing_acc{testing_acc}')

    # import pandas as pd
    # data = pd.DataFrame()
    # data['class_no'] = list(range(args.num_class))
    # data['train_acc'] = training_acc_list_all_class
    # data['test_acc'] = testing_acc_list_all_class
    #
    # data.to_csv(r"checkpoint/densenetbc/Saturday_06_March_2021_14h_25m_48s/Train_test_acc.csv", mode='a', index=False)
