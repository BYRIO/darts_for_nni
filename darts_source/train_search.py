"""
File: train_search.py
Author: OccamRazorTeam
Email: vincent.duan95@outlook.com
Github: https://github.com/VDeamoV
Description: This file is used to search model architect, 
    - one folder: the scripts for future use
    - one log: contain the training result
    - one weight.pth: contain the train weights
"""

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
import nni

import utils
import model_search_parser
from model_search import Network, tuner_params
from architect import Architect
from custom_visualize import plot
from data.TorchDatasetLoader.base_dataset import CustomImageDataset


# get params from files
args = model_search_parser.get_cifar_parser_params()


# get where to save log
args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(os.path.join(tuner_params['output_path'], args.save), scripts_to_save=glob.glob('*.py'))


# create log
# basic settings
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
# create where to save log
log_file = logging.FileHandler(os.path.join(tuner_params['output_path'],args.save,'log.txt'))
log_file.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(log_file)


def main():
    # Information Output
    # Check GPU
    if not torch.cuda.is_available():
        logging.info('NO GPU DEVICE AVAILABLE')
        sys.exit(1)
    logging.info("%s", tuner_params["dataset_path"])
    logging.info("Model Params = %s", args)
    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    logging.info('gpu device = %d' % args.gpu)

    # Advanced Cudnn init
    cudnn.benchmark = True
    cudnn.enabled = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


    # Dataset & Dataloader init
    DATASET_CONFIG_PATH = "/home/apex/DeamoV/github/darts_for_nni/custom_mini_imagenet.yaml"
    train_data = CustomImageDataset(DATASET_CONFIG_PATH, mode='train', debug=False)
    CIFAR_CLASSES = train_data.num_train_classes
    #  TODO: need to fix CIFAR_CLASSES # 
    
    # CIFAR_CLASSES = 10
    # train_transform, valid_transform = utils._data_transforms_cifar10(args)
    # train_data = dset.CIFAR10(
    #     root=tuner_params["dataset_path"],
    #     train=True,
    #     download=True,
    #     transform=train_transform)

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))

    train_dataloader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True,
        num_workers=2)

    valid_dataloader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=True,
        num_workers=2)

    # init loss
    criterion_loss = nn.CrossEntropyLoss()
    criterion_loss = criterion_loss.cuda()

    # init model & init architect
    model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion_loss)
    model = model.cuda()
    architect = Architect(model, args)
    logging.info("Model Param size = %fMB", utils.count_parameters(model))

    # optim & scheduler
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        float(args.epochs),
        eta_min=args.learning_rate_min)

    # train_loop
    for epoch in range(args.epochs):
        scheduler.step()
        lr = scheduler.get_lr()[0]

        genotype = model.genotype()
        genotypes_py = open(os.path.join(tuner_params['output_path'],args.save)
                + "/scripts/genotypes.py","a")
        genotypes_py.write("DARTS = " + str(genotype))
        genotypes_py.close()

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
        
        logging.info("start plotting epoch:" + str(epoch))
        plot(genotype.normal, "normal" + "_epoch_" + str(epoch) + "_valid_acc_" + str(valid_acc) + "_train_acc_" + str(train_acc), os.path.join(tuner_params['output_path'], args.save))
        plot(genotype.reduce, "reduce" + "_epoch_" + str(epoch) + "_valid_acc_" + str(valid_acc) + "_train_acc_" + str(train_acc), os.path.join(tuner_params['output_path'], args.save))
        logging.info("end plotting")

        utils.save(model, os.path.join(tuner_params['output_path'], args.save, 'weights.pt'))


def train(train_dataloader, valid_dataloader, model, architect, criterion_loss, optimizer, lr):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    for step, (input_data, target) in enumerate(train_dataloader):
        model.train()
        n = input_data.size(0)

        input_data = input_data.cuda()
        target = target.cuda(async=True)

        # get a random minibatch from the search queue with replacement
        input_search, target_search = next(iter(valid_dataloader))
        input_search = input_search.cuda()
        target_search = target_search.cuda(async=True)

        architect.step(input_data, target, input_search, target_search,
                       lr, optimizer, unrolled=args.unrolled)

        optimizer.zero_grad()
        logits = model(input_data)
        loss = criterion_loss(logits, target)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step % args.report_freq == 0:
            logging.info('train step:%03d loss:%e top1:%f top5:%f', 
                         step, objs.avg, top1.avg, top5.avg)

        nni.report_intermediate_result(objs.avg)

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
            logging.info('valid step:%03d loss:%e top1:%f top5:%f', step,
                         objs.avg, top1.avg, top5.avg)

    return top5.avg, objs.avg


if __name__ == '__main__':
    main()
