import os
import argparse


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import core.datasets as datasets
import core.losses as losses
from core.utils.warp import warp3D
from core.framework import Framework, init


def fetch_loss(affines, deforms, agg_flow, image1, image2, AGG_flows):
    det = losses.det3x3(affines['A'])
    det_loss = torch.sum((det - 1.0) ** 2) / 2

    I = torch.cuda.FloatTensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]])
    eps = 1e-5
    epsI = torch.cuda.FloatTensor([[[eps * elem for elem in row] for row in Mat] for Mat in I])
    C = torch.matmul(affines['A'].permute(0, 2, 1), affines['A']) + epsI
    s1, s2, s3 = losses.elem_sym_polys_of_eigen_values(C)
    ortho_loss = torch.sum(s1 + (1 + eps) * (1 + eps) * s2 / s3 - 3 * 2 * (1 + eps))

    image2_warped = warp3D()(image2, agg_flow)
    sim_loss = 0.1 * det_loss + 0.1 * ortho_loss + losses.similarity_loss(image1, image2_warped)

    reg_loss = 0.0
    for i in range(len(deforms)):
        reg_loss = reg_loss + losses.regularize_loss(deforms[i]['flow'])

    pure_loss = sim_loss + 0.5 * reg_loss

    dist_loss = losses.distill_loss(AGG_flows)

    whole_loss = pure_loss + 0.1 * dist_loss

    metrics = {'loss': pure_loss.item()}

    return [pure_loss, whole_loss], metrics


def fetch_dataloader(args):
    if args.dataset == 'liver':
        train_dataset = datasets.LiverTrain(args)
    elif args.dataset == 'brain':
        train_dataset = datasets.BrainTrain(args)
    elif args.dataset == 'oasis':
        train_dataset = datasets.OasisTrain(args)
    elif args.dataset == 'mindboggle':
        train_dataset = datasets.MindboggleTrain(args)
    else:
        print('Wrong Dataset')

    gpuargs = {'num_workers': 4, 'drop_last': True}
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.batch, pin_memory=True, shuffle=False, sampler=train_sampler, **gpuargs)

    if args.local_rank == 0:
        print('Image pairs in training: %d' % len(train_dataset), file=args.files, flush=True)
    return train_loader


def fetch_optimizer(args, model):
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    milestones = [args.round*3, args.round*4, args.round*5]  # args.epoch == 5
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)

    return optimizer, scheduler


class Logger:
    def __init__(self, model, scheduler, args):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.sum_freq = args.sum_freq

    def _print_training_status(self):
        metrics_data = ["{" + k + ":{:10.5f}".format(self.running_loss[k] / self.sum_freq) + "} "
                        for k in self.running_loss.keys()]
        training_str = "[Steps:{:9d}, Lr:{:10.7f}] ".format(self.total_steps + 1, self.scheduler.get_lr()[0])
        print(training_str + "".join(metrics_data), file=args.files, flush=True)
        print(training_str + "".join(metrics_data))

        for key in self.running_loss:
            self.running_loss[key] = 0.0

    def push(self, metrics):
        self.total_steps = self.total_steps + 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] = self.running_loss[key] + metrics[key]

        if self.total_steps % self.sum_freq == self.sum_freq - 1:
            if args.local_rank == 0:
                self._print_training_status()
            self.running_loss = {}


def train(args):
    model = Framework(args)
    model.cuda()

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)

    train_loader = fetch_dataloader(args)
    optimizer, scheduler = fetch_optimizer(args, model)

    total_steps = 0
    logger = Logger(model, scheduler, args)

    should_keep_training = True
    while should_keep_training:
        for i_batch, data_blob in enumerate(train_loader):
            model.train()
            image1, image2 = [x.cuda(non_blocking=True) for x in data_blob]

            optimizer.zero_grad()
            image2_aug, affines, deforms, agg_flow, AGG_flows = model(image1, image2)

            losses, metrics = fetch_loss(affines, deforms, agg_flow, image1, image2_aug, AGG_flows)

            total_steps = total_steps + 1
            logger.push(metrics)
            loss = losses[0] if total_steps > 5*args.round else losses[1]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            scheduler.step()

            if total_steps % args.val_freq == args.val_freq - 1:
                PATH = args.model_path + '/%s_%d.pth' % (args.name, total_steps + 1)
                torch.save(model.state_dict(), PATH)

            if total_steps == args.num_steps:
                should_keep_training = False
                break

    PATH = args.model_path + '/%s.pth' % args.name
    torch.save(model.state_dict(), PATH)

    return PATH


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='SDHNet_UpLoad', help='name your experiment')
    parser.add_argument('--dataset', type=str, default='brain', help='which dataset to use for training')
    parser.add_argument('--epoch', type=int, default=5, help='number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--batch', type=int, default=1, help='number of image pairs per batch on single gpu')
    parser.add_argument('--sum_freq', type=int, default=1000)
    parser.add_argument('--val_freq', type=int, default=2000)
    parser.add_argument('--round', type=int, default=20000, help='number of batches per epoch')
    parser.add_argument('--data_path', type=str, default='E:/Registration/Code/TMI2022/Github/Data_MRIBrain/')
    parser.add_argument('--base_path', type=str, default='E:/Registration/Code/TMI2022/Github/')
    parser.add_argument('--iters', type=int, default=6)
    parser.add_argument('--local_rank', default=0, type=int, help='node rank for distributed training')
    args = parser.parse_args()

    args.model_path = args.base_path + args.name + '/output/checkpoints_' + args.dataset
    os.makedirs(args.model_path, exist_ok=True)
    args.eval_path = args.base_path + args.name + '/output/eval_' + args.dataset
    os.makedirs(args.eval_path, exist_ok=True)

    if args.local_rank == 0:
        init(args)

    args.nums_gpu = torch.cuda.device_count()
    args.batch = args.batch
    args.num_steps = args.epoch * args.round
    args.files = open(args.base_path + args.name + '/output/train_' + args.dataset + '.txt', 'a+')

    if args.local_rank == 0:
        print('Dataset: %s' % args.dataset, file=args.files, flush=True)
        print('Batch size: %s' % args.batch, file=args.files, flush=True)
        print('Step: %s' % args.num_steps, file=args.files, flush=True)
        print('Parallel GPU: %s' % args.nums_gpu, file=args.files, flush=True)

    train(args)
    args.files.close()
