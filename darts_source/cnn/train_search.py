import os
import sys
import time
import glob
import argparse
import logging


import torch
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import numpy as np
import tqdm
import nni


# noinspection PyUnresolvedReferences
from model_search import Network
from architect import Architect
import utils
import model_parser

# get params from files
args = parser.get_parser_params()

# get where to save log
args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save)


# create log
# basic settings
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
log_format = '%(asctime)s %(message)s'
# create where to save log
log_file = logging.FileHandler(os.path.join(args.save, 'log.txt'))
log_file.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(log_file)


CIFAR_CLASSES = 10


def main():
    # Information Output
    # check gpu
    if not torch.cuda.is_available():
        logging.info('NO GPU DEVICE AVAILABLE')
        sys.exit(1)
    logging.info("Model Params = %s", args)
    np.random.seed(args.seed)
    # want use multi gpu
    # torch.cuda.set_device(args.gpu)
    # logging.info('gpu device = %d' % args.gpu)

    cudnn.benchmark = True
    cudnn.enabled = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    criterion_loss = nn.CrossEntropyLoss()
    criterion_loss = criterion_loss.cuda()
    model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion_loss)

    # enable multi gpu
    model = nn.DataParallel(model)
    model = model.cuda()

    logging.info("Model Param size = %fMB", utils.count_parameters(model))

    optimizer = torch.optim.SGD(model.parameters(),
                                args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    train_transform, valid_transform = utils._data_transforms_cifar10(args)
    train_data = dset.CIFAR10(
            root=args.data, train=True, download=True,
            transform=train_transform)

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))

    train_dataloader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True, num_workers=2)

    valid_dataloader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(
            indices[split:num_train]),
        pin_memory=True, num_workers=2)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)

    architect = Architect(model, args)

    for epoch in tqdm(range(args.epochs)):
        scheduler.step()
        lr = scheduler.get_lr()[0]

        genotype = model.genotype()
        logging.info('epoch %d lr %e', epoch, lr)
        logging.info('genotype = %s', genotype)

        print(F.softmax(model.alphas_normal, dim=-1))
        print(F.softmax(model.alphas_reduce, dim=-1))

        # training
        train_acc, train_obj = train(train_dataloader, valid_dataloader, model, architect, criterion_loss, optimizer, lr)

        # validation
        valid_acc, valid_obj = val(valid_dataloader, model, criterion_loss)
        logging.info('train_acc %f', train_acc)
        logging.info('valid_acc %f', valid_acc)

        utils.save(model, os.path.join(args.save, 'weights.pt'))


def train(train_dataloader, valid_dataloader, model, architect, criterion_loss, optimizer, lr):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    for step, (input, target) in enumerate(train_dataloader):
        model.train()
        n = input.size(0)

        input = Variable(input, requires_grad=False).cuda()
        target = Variable(target, requires_grad=False).cuda(async=True)

        # get a random minibatch from the search queue with replacement
        input_search, target_search = next(iter(valid_dataloader))
        input_search = Variable(input_search, requires_grad=False).cuda()
        target_search = Variable(
            target_search, requires_grad=False).cuda(async=True)

        architect.step(input, target, input_search, target_search,
                       lr, optimizer, unrolled=args.unrolled)

        optimizer.zero_grad()
        logits = model(input)
        loss = criterion_loss(logits, target)

        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step % args.report_freq == 0:
            logging.info('train %03d %e %f %f', step,
                         objs.avg, top1.avg, top5.avg)

    return top5.avg, objs.avg


def val(valid_dataloader, model, criterion_loss):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    for step, (input, target) in enumerate(valid_dataloader):
        with torch.no_grad():
            input = Variable(input).cuda()
            target = Variable(target).cuda(async=True)

        logits = model(input)
        loss = criterion_loss(logits, target)

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step % args.report_freq == 0:
            logging.info('valid %03d %e %f %f', step,
                         objs.avg, top1.avg, top5.avg)

    return top5.avg, objs.avg


if __name__ == '__main__':
    main()
