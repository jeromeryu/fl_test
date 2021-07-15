#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import copy
import os
import shutil
import warnings

import numpy as np
from tqdm import tqdm
from thop import profile, clever_format

warnings.filterwarnings('ignore')
import torch
# from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate
from models import CNNMnist, CNNCifar, Model
from utils import get_dataset, average_weights, exp_details
import torch.nn.functional as F



# test for one epoch, use weighted knn to find the most similar images' label to assign the test image
def test(net, memory_data_loader, test_data_loader):
    net.eval()
    total_top1, total_top5, total_num, feature_bank, target_bank = 0.0, 0.0, 0, [], []
    with torch.no_grad():
        # generate feature bank and target bank
        for data_tuple in tqdm(memory_data_loader, desc='Feature extracting'):
            (data, _), target = data_tuple
            target_bank.append(target)
            feature, out = net(data.cuda(non_blocking=True))
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.cat(target_bank, dim=0).contiguous().to(feature_bank.device)
        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader)
        for data_tuple in test_bar:
            (data, _), target = data_tuple
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            feature, out = net(data)

            total_num += data.size(0)
            # compute cos similarity between each feature vector and feature bank ---> [B, N]
            sim_matrix = torch.mm(feature, feature_bank)
            # [B, K]
            sim_weight, sim_indices = sim_matrix.topk(k=200, dim=-1) # k = 200
            # [B, K]
            sim_labels = torch.gather(feature_labels.expand(data.size(0), -1), dim=-1, index=sim_indices)
            sim_weight = (sim_weight / 0.5).exp() #temperature = 0.5

            # counts for each class
            one_hot_label = torch.zeros(data.size(0) * 200, c, device=sim_labels.device) #k=200, 
            # [B*K, C]
            one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
            # weighted score ---> [B, C]
            pred_scores = torch.sum(one_hot_label.view(data.size(0), -1, c) * sim_weight.unsqueeze(dim=-1), dim=1)

            pred_labels = pred_scores.argsort(dim=-1, descending=True)
            total_top1 += torch.sum((pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_top5 += torch.sum((pred_labels[:, :5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            test_bar.set_description('Test Epoch: [{}/{}] Acc@1:{:.2f}% Acc@5:{:.2f}%'
                                     .format(epoch, args.epochs, total_top1 / total_num * 100, total_top5 / total_num * 100))

    return total_top1 / total_num * 100, total_top5 / total_num * 100


def inference(model, test_loader):
    model.eval()
    test_loss = 0.0
    correct = 0.0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            test_loss += F.cross_entropy(output, target).item()
            pred = torch.max(output, 1)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)
    return acc, test_loss


def prepare_folders(cur_path):
    folders_util = [
        os.path.join(cur_path , 'logs', args.store_name),
        os.path.join(cur_path , 'checkpoints', args.store_name)]
    for folder in folders_util:
        if not os.path.exists(folder):
            print('creating folder ' + folder)
            # os.mkdir(folder)
            os.makedirs(folder)


def save_checkpoint(state, is_best):
    filename = '{}/{}/ckpt.pth.tar'.format(os.path.abspath(os.path.dirname(os.getcwd())) + '/checkpoints',
                                           args.store_name)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('pth.tar', 'best.pth.tar'))


if __name__ == '__main__':

    args = args_parser()
    args.type = 'iid' if args.iid == 1 else 'non-iid'
    args.store_name = '_'.join(
        [args.dataset, args.model, args.type, 'lr-' + str(args.lr)])
    cur_path = os.path.abspath(os.path.dirname(os.getcwd()))
    prepare_folders(cur_path)
    exp_details(args)

    GPU_NUM = 1
    device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device) # change allocation of current GPU


    logger_file = open(os.path.join(cur_path + '/logs', args.store_name, 'log.txt'), 'w')
    # tf_writer = SummaryWriter(log_dir=os.path.join(cur_path + '/logs', args.store_name))

    # load dataset and user groups
    train_dataset, memory_dataset, test_dataset, user_groups= get_dataset(args)
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128,
    #                                           shuffle=False, num_workers=4)

    memory_loader = torch.utils.data.DataLoader(memory_dataset, batch_size=512, shuffle=False,
            num_workers=16, pin_memory=False)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=512,shuffle=False, 
                num_workers=16, pin_memory=False)
    c = len(memory_dataset.classes)

    # BUILD MODEL
    # if args.dataset == 'mnist':
        # global_model = CNNMnist(args).cuda()
    # elif args.dataset == 'cifar':
    #     global_model = CNNCifar(args).cuda()

    global_model = Model().cuda()
    flops, params = profile(global_model, inputs=(torch.randn(1, 3, 32, 32).cuda(),))

    bst_acc = -1
    description = "inference acc={:.4f}% loss={:.2f}, best_acc = {:.2f}%"
    for epoch in tqdm(range(args.epochs)):
    # for epoch in range(args.epochs):
        local_weights = []
        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for idx in idxs_users:
            # print("idx", idx)
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx])
            # w = local_model.update_weights(
            #     model=copy.deepcopy(global_model))
            w = local_model.train(net = copy.deepcopy(global_model))
            local_weights.append(copy.deepcopy(w))

        # update global weights
        global_weights = average_weights(local_weights)


        # update global weights
        global_model.load_state_dict(global_weights)

        # test_acc, test_loss = inference(global_model, test_loader)

        test_acc_1, test_acc_5 = test(global_model, memory_loader, test_loader)


        # tf_writer.add_scalar('test_acc', test_acc, epoch)
        # tf_writer.add_scalar('test_loss', test_loss, epoch)

        # output_log = 'After {} global rounds, Test acc: {}, inference loss: {}'.format(
        #     epoch + 1, test_acc, test_loss)



        output_log = 'After {} global rounds, Test acc1: {}, Test acc5: {}'.format(
            epoch + 1, test_acc_1, test_acc_5)
        
        
        logger_file.write(output_log + '\n')
        logger_file.flush()


        # is_best = test_acc > bst_acc
        # bst_acc = max(bst_acc, test_acc)
        # print(description.format(test_acc, test_loss, bst_acc))
        
        # save_checkpoint(global_model.state_dict(), is_best)

"""
python federated_main.py --model=cnn --dataset=cifar --iid=1 --epochs=300 --lr=0.01 --local_ep=5 --local_bs=32

python3 federated_main.py --model=cnn --dataset=mnist --iid=1 --epochs=100 --lr=0.01 --local_ep=5 --local_bs=32

"""