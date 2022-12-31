import argparse
import torch, os
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms
from models import get_model
import random
import numpy as np
from glob import glob
from PIL import Image
import time
#from progress import bar
from utils import Bar, Logger, AverageMeter, accuracy, savefig
import shutil
import json
from pprint import pprint
import argparse
import logging
import os
import sys

from data_loader_celeba import load_partition_data_Celeba
from fedml_api.model.linear.lr import LogisticRegression

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))
from fedml_api.centralized.centralized_trainer import CentralizedTrainer

from fedml_api.model.cv.resnet_gn import resnet50
from torchvision.models import resnet18
import fedml
import torch
from fedml.simulation import SimulatorSingleProcess as Simulator
from fedml import FEDML_TRAINING_PLATFORM_SIMULATION, FEDML_SIMULATION_TYPE_SP

fedml._globazl_training_type = FEDML_TRAINING_PLATFORM_SIMULATION
backend=FEDML_SIMULATION_TYPE_SP
fedml._global_comm_backend = backend


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='imagenet_checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'imagenet_model_best.pth.tar'))

#调整学习率
def adjust_learning_rate(lr, optimizer, epoch, args):
    # global state
    # lr = args.lr
    if epoch in args.schedule:
        lr *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return lr


class bd_data(data.Dataset):
    def __init__(self, data_dir, bd_label, mode, transform, bd_ratio):
        self.bd_list = glob(data_dir + '/' + mode + '/*_hidden*')
        self.transform = transform
        self.bd_label = bd_label
        self.bd_ratio = bd_ratio  # since all bd data are 0.1 of original data, so ratio = bd_ratio / 0.1

        n = int(len(self.bd_list) * (bd_ratio / 0.1))
        self.bd_list = self.bd_list[:n]

    def __len__(self):
        return len(self.bd_list)

    def __getitem__(self, item):
        im = Image.open(self.bd_list[item])
        if self.transform:
            input = self.transform(im)
        else:
            input = np.array(im)
        
        return input, self.bd_label


class bd_data_val(data.Dataset):
    def __init__(self, data_dir, bd_label, mode, transform, label_index_list):
        self.bd_list = glob(data_dir + '/' + mode + '/*_hidden*')
        self.bd_list = [item for item in self.bd_list if label_index_list[bd_label] not in item]
        self.transform = transform
        self.bd_label = bd_label
        
    def __len__(self):
        return len(self.bd_list)

    def __getitem__(self, item):
        im = Image.open(self.bd_list[item])
        if self.transform:
            input = self.transform(im)
        else:
            input = np.array(im)
        
        return input, self.bd_label

def train(model, dataloader, bd_dataloader, criterion, optimizer, use_cuda):
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(dataloader))
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        # measure data loading time
        inputs_trigger, targets_trigger = bd_dataloader.__iter__().__next__()
        inputs = torch.cat((inputs, inputs_trigger), 0)
        targets = torch.cat((targets, targets_trigger), 0)
        
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(dataloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)


def test(model, testloader, criterion, use_cuda):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar('Processing', max=len(testloader))
    for batch_idx, (inputs, targets) in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(testloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)



def main(args):
    pprint(args.__dict__)

    if not os.path.exists(args.checkpoint):
        os.makedirs(args.checkpoint)

    # Save arguments into txt
    with open(os.path.join(args.checkpoint, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=4)

    best_acc_clean = 0
    best_acc_trigger = 0
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch
    
    title = 'training bd imagenet'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    use_cuda = torch.cuda.is_available()

    # Random seed
    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.manualSeed)

    batch_size_org = int(round(args.train_batch * (1 - 0.1)))
    batch_size_bd = args.train_batch - batch_size_org

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
        ]),
        'test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
        ])
    }
    image_datasets = {x: datasets.ImageFolder(os.path.join(args.data_dir, x), data_transforms[x]) 
                    for x in ['train', 'val','test']}
    train_loader = data.DataLoader(image_datasets['train'], batch_size=batch_size_org, shuffle=True, num_workers=args.workers)
    val_loader = data.DataLoader(image_datasets['val'], batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
    

    bd_image_datasets = {x: bd_data(args.bd_data_dir, args.bd_label, x, data_transforms[x], args.bd_ratio) for x in ['train', 'val']}
    bd_train_loader = data.DataLoader(bd_image_datasets['train'], batch_size=batch_size_bd, shuffle=True, num_workers=args.workers)
    
    label_index_list = sorted(os.listdir(args.data_dir + '/val'))
    bd_image_datasets_val = bd_data_val(args.bd_data_dir, args.bd_label, 'val', data_transforms['val'], label_index_list)
    bd_val_loader = data.DataLoader(bd_image_datasets_val, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

    # Selecting models
    model = get_model(args.net)    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    #Loss Function
    criterion = nn.CrossEntropyLoss()
    # Observe that all parameters are being optimized
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # if not os.path.exists(args.checkpoint):
    #     os.makedirs(args.checkpoint)

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        # assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        try:
            # args.checkpoint = os.path.dirname(args.resume)
            checkpoint = torch.load(args.resume)
            best_acc_clean = checkpoint['best_acc_clean']
            best_acc_trigger = checkpoint['best_acc_trigger']
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger = Logger(os.path.join(args.checkpoint, 'imagenet.txt'), title=title, resume=True)
        except:
            logger = Logger(os.path.join(args.checkpoint, 'imagenet.txt'), title=title)
            logger.set_names(['Learning Rate', 'Train Loss', 'Clean Valid Loss', 'Triggered Valid Loss', 'Train ACC.', 'Valid ACC.', 'ASR'])
    else:
        logger = Logger(os.path.join(args.checkpoint, 'imagenet.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Clean Valid Loss', 'Triggered Valid Loss', 'Train ACC.', 'Valid ACC.', 'ASR'])
    
    # Train and val
    lr = args.lr
    for epoch in range(start_epoch, args.epochs):
        lr = adjust_learning_rate(lr, optimizer, epoch, args) 
    
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, lr))
    
        train_loss, train_acc = train(model, train_loader, bd_train_loader, criterion, optimizer, use_cuda)
        test_loss_clean, test_acc_clean = test(model, val_loader, criterion, use_cuda)
        test_loss_trigger, test_acc_trigger = test(model, bd_val_loader, criterion, use_cuda)
    
        # append logger file
        logger.append([lr, train_loss, test_loss_clean, test_loss_trigger, train_acc, test_acc_clean, test_acc_trigger])
    
        # save model
        is_best = (test_acc_clean + test_acc_trigger) > (best_acc_clean + best_acc_trigger)
        if is_best:
            best_acc_clean = test_acc_clean
            best_acc_trigger = test_acc_trigger
            
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc_clean': test_acc_clean,
                'acc_trigger': test_acc_trigger,
                'best_acc_clean': best_acc_clean,
                'best_acc_trigger': best_acc_trigger,
                'optimizer' : optimizer.state_dict(),
            }, is_best, checkpoint=args.checkpoint)
    
    
    logger.close()
    logger.plot()
    # savefig(os.path.join(args.checkpoint, 'imagenet.eps'))
    
    print('Best accs (clean,trigger):')
    print(best_acc_clean, best_acc_trigger)

def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument('--model', type=str, default='logsr', metavar='N',
                        help='neural network used in training')

    parser.add_argument('--DP_if', type=str, default=0,
                        help='U-LDP use')

    #换成celeba
    parser.add_argument('--dataset', type=str, default='celeba', metavar='N',
                        help='dataset used for training')
    #数据集目录
    parser.add_argument('--data_dir', type=str, default='./../../../data/mnist',
                        help='data directory')
    #怎么区分开本地数据集
    parser.add_argument('--partition_method', type=str, default='hetero', metavar='N',
                        help='how to partition the dataset on local workers')

    parser.add_argument('--partition_alpha', type=float, default=0.5, metavar='PA',
                        help='partition alpha (default: 0.5)')
    #一共多少客户端
    parser.add_argument('--client_num_in_total', type=int, default=10, metavar='NN',
                        help='number of workers in a distributed cluster')
    #每一轮有多少客户端参与
    parser.add_argument('--client_num_per_round', type=int, default=10, metavar='NN',
                        help='number of workers')

    parser.add_argument('--batch_size', type=int, default=10, metavar='N',
                        help='input batch size for training (default: 64)')
    #客户端训练的优化器
    parser.add_argument('--client_optimizer', type=str, default='adam',
                        help='SGD with momentum; adam')

    #FL的训练的优化器
    parser.add_argument('--federated_optimizer', type=str, default='FedAvg',
                        help='FedAvg')

    parser.add_argument('--learning_rate', type=float, default=0.001, metavar='lr',
                        help='learning rate (default: 0.001)')

    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')

    parser.add_argument('--wd', type=float, default=0.001, metavar='wd',
                        help='weight_decay (default: 0.001)')

    parser.add_argument('--weight_decay', help='weight decay parameter;', type=float, default=0.001)


    parser.add_argument('--epochs', type=int, default=5, metavar='EP',
                        help='how many epochs will be trained locally')

    parser.add_argument('--comm_round', type=int, default=10,
                        help='how many round of communications we shoud use')

    parser.add_argument('--is_mobile', type=int, default=0,
                        help='whether the program is running on the FedML-Mobile server side')

    parser.add_argument('--frequency_of_train_acc_report', type=int, default=1,
                        help='the frequency of training accuracy report')

    parser.add_argument('--frequency_of_test_acc_report', type=int, default=1,
                        help='the frequency of test accuracy report')

    parser.add_argument('--gpu_server_num', type=int, default=1,
                        help='gpu_server_num')

    parser.add_argument('--frequency_of_the_test', type=int, default=5,
                        help='frequency_of_the_test')

    parser.add_argument('--gpu_num_per_server', type=int, default=4,
                        help='gpu_num_per_server')

    parser.add_argument('--ci', type=int, default=0,
                        help='CI')

    parser.add_argument('--gpu', type=int, default=0,
                        help='gpu')

    parser.add_argument('--poison_num', type=int, default=1600,
                        help='poison_temple_number')

    #federated_optimizer: "FedAvg"
    args = parser.parse_args()
    return args



def load_data(args,dataset_name):
    logging.info("load_data. dataset_name = %s" % dataset_name)

    client_num,train_data_num, test_data_num, train_data_global, test_data_global, \
    train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
    class_num = load_partition_data_Celeba(args,batch_size=64)
    args.client_num_in_total = 10
    dataset = [train_data_num, test_data_num, train_data_global, test_data_global,
               train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num]
    return dataset



def create_model(args,model_name,output_dim):
    logging.info("create_model. model_name = %s, output_dim = %s" % (model_name, output_dim))
    model = None
    if model_name == 'logsr':
        model = resnet18() #因为是2分类问题
    return model



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    args = add_args(parser)
    logging.info(args)
    dataset = load_data(args, args.dataset)


    model = create_model(args, model_name=args.model, output_dim=2)


    device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")

    single_trainer = CentralizedTrainer(dataset, model, device, args)
    single_trainer.train()


