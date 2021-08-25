#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, dataset
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        # image, label = self.dataset[self.idxs[item]]
        # return torch.tensor(image), torch.tensor(label)
        return self.dataset[self.idxs[item]]


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs):
        self.args = args
        self.dataset = DatasetSplit(dataset, idxs)
        self.trainloader = DataLoader(self.dataset,
                            batch_size=self.args.local_bs, shuffle=True, num_workers=4, pin_memory=False, drop_last=True)


    def update_weights(self, model):
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                    momentum=0.9)

        for iter in range(self.args.local_ep):
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.cuda(), labels.cuda()
                model.zero_grad()
                output = model(images)
                loss = F.cross_entropy(output, labels)
                loss.backward() 
                optimizer.step()
        return model.state_dict()


    def train(self, net, batch_size):
        train_optimizer = optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-6)
        net.train()
        # total_loss, total_num, train_bar = 0.0, 0, tqdm(self.trainloader)
        for iter in range(self.args.local_ep):
            total_loss, total_num, train_bar = 0.0, 0, tqdm(self.trainloader)
            for data_tuple in train_bar:
                (pos_1, pos_2), _ = data_tuple
                pos_1, pos_2 = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True)
                feature_1, out_1 = net(pos_1)
                feature_2, out_2 = net(pos_2)
                # Barlow Twins
                
                print("p1", out_1)
                print("p2", out_2)
                
                # normalize the representations along the batch dimension
                out_1_norm = (out_1 - out_1.mean(dim=0)) / out_1.std(dim=0)
                out_2_norm = (out_2 - out_2.mean(dim=0)) / out_2.std(dim=0)
                
                # cross-correlation matrix
                c = torch.matmul(out_1_norm.T, out_2_norm) / batch_size #batch_size

                # loss
                on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
                # if corr_neg_one is False:
                #     # the loss described in the original Barlow Twin's paper
                #     # encouraging off_diag to be zero
                off_diag = off_diagonal(c).pow_(2).sum()
                # else:
                #     # inspired by HSIC
                #     # encouraging off_diag to be negative ones
                #     off_diag = off_diagonal(c).add_(1).pow_(2).sum()
                loss = on_diag + 0.0078125 * off_diag
                print("loss", loss)                

                train_optimizer.zero_grad()
                loss.backward()
                train_optimizer.step()

                total_num += batch_size #batch_size
                total_loss += loss.item() * batch_size #batch_size
                # if corr_neg_one is True:
                #     off_corr = -1
                # else:
                off_corr = 0
                train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f} bsz:{} len:{} '.format(
                                        iter, self.args.local_ep , total_loss / total_num, batch_size, len(self.dataset)))
            # return total_loss / total_num
        return net.state_dict()