#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

#argparse是一个Python模块：命令行选项、参数和子命令解析器。
#主要有三个步骤：
#创建 ArgumentParser() 对象
#调用 add_argument() 方法添加参数
#使用 parse_args() 解析添加的参数
import argparse


def args_parser():
    #
    parser = argparse.ArgumentParser()
    # 联邦学习的参数
    #type - 命令行参数应该被转换成的类型。     default - 不指定参数时的默认值    help - 参数的帮助信息
    parser.add_argument('--epochs', type=int, default=10, help="rounds of training")   #训练轮数，是指全局迭代次数吗？
    parser.add_argument('--num_users', type=int, default=100, help="number of users: K")#每一轮迭代时，服务器选择的参与训练的客户端的数量
    parser.add_argument('--frac', type=float, default=0.1, help='the fraction of clients: C')#客户端所占的比例
    parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E")#本地的迭代次数
    parser.add_argument('--local_bs', type=int, default=10, help="local batch size: B")#本地训练时每一轮的样本数
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')#学习率
    parser.add_argument('--momentum', type=float, default=0.5, help='SGD momentum (default: 0.5)')

    # model arguments   模型参数
    parser.add_argument('--model', type=str, default='mlp', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9, help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to use for convolution')
    parser.add_argument('--norm', type=str, default='batch_norm', help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32,
                        help="number of filters for conv nets -- 32 for miniimagenet, 64 for omiglot.")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than strided convolutions")

    # other arguments   其他参数
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    parser.add_argument('--iid', type=int, default=0, help='whether i.i.d or not, 1 for iid, 0 for non-iid')
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=3, help="number of channels of imges")
    parser.add_argument('--gpu', type=int, default=-1, help="GPU ID")
    parser.add_argument('--stopping_rounds', type=int, default=10, help='rounds of early stopping')
    parser.add_argument('--verbose', type=int, default=1, help='verbose print, 1 for True, 0 for False')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--ag_scalar', type=float, default=1.0, help="global aggregation updating scalar, simplicity for A Matrix")
    parser.add_argument('--lg_scalar', type=float, default=1.0, help="client local updating scalar, simplicity for S Matrix")
    parser.add_argument('--algorithm', type=str, default='fedavg', help='algorithm for optimization')
    args = parser.parse_args()
    return args

