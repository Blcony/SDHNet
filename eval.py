import os
import numpy as np
from os.path import join
import argparse

import torch
import torch.nn as nn

import core.datasets as datasets
from core.utils.warp import warp3D
from core.framework import Framework


def mask_class(label, value):
    return (torch.abs(label - value) < 0.5).float() * 255.0


def mask_metrics(seg1, seg2):
    sizes = np.prod(seg1.shape[1:])
    seg1 = (seg1.view(-1, sizes) > 128).type(torch.float32)
    seg2 = (seg2.view(-1, sizes) > 128).type(torch.float32)
    dice_score = 2.0 * torch.sum(seg1 * seg2, 1) / (torch.sum(seg1, 1) + torch.sum(seg2, 1))

    union = torch.sum(torch.max(seg1, seg2), 1)
    iden = (torch.ones(*union.shape) * 0.01).cuda()
    jacc_score = torch.sum(torch.min(seg1, seg2), 1) / torch.max(iden, union)

    return dice_score, jacc_score


def jacobian_det(flow):
    bias_d = np.array([0, 0, 1])
    bias_h = np.array([0, 1, 0])
    bias_w = np.array([1, 0, 0])

    volume_d = np.transpose(flow[:, 1:, :-1, :-1] - flow[:, :-1, :-1, :-1], (1, 2, 3, 0)) + bias_d
    volume_h = np.transpose(flow[:, :-1, 1:, :-1] - flow[:, :-1, :-1, :-1], (1, 2, 3, 0)) + bias_h
    volume_w = np.transpose(flow[:, :-1, :-1, 1:] - flow[:, :-1, :-1, :-1], (1, 2, 3, 0)) + bias_w

    jacobian_det_volume = np.linalg.det(np.stack([volume_w, volume_h, volume_d], -1))
    jd = np.sum(jacobian_det_volume <= 0)
    return jd


def evaluate_liver(args, model, steps):
    for datas in args.dataset_test:
        eval_path = join(args.eval_path, datas)
        if (args.local_rank == 0) and (not os.path.isdir(eval_path)):
            os.makedirs(eval_path)
        file_sum = join(eval_path, datas + '.txt')
        file = join(eval_path, datas + '_' + str(steps) + '.txt')
        f = open(file, 'a+')
        g = open(file_sum, 'a+')

        Dice, Jacc, Jacb = [], [], []
        if 'lspig' in datas:
            eval_dataset = datasets.LspigTest(args, datas)
        else:
            eval_dataset = datasets.LiverTest(args, datas)
        if args.local_rank == 0:
            print('Dataset in evaluation: %s' % datas, file=f)
            print('Image pairs in evaluation: %d' % len(eval_dataset), file=f)
            print('Evaluation steps: %s' % steps, file=f)

        for i in range(len(eval_dataset)):
            image1, image2 = eval_dataset[i][0][np.newaxis].cuda(), eval_dataset[i][1][np.newaxis].cuda()
            label1, label2 = eval_dataset[i][2][np.newaxis].cuda(), eval_dataset[i][3][np.newaxis].cuda()

            with torch.no_grad():
                _, _, _, agg_flow, _ = model.module(image1, image2, augment=False)
                label2_warped = warp3D()(label2, agg_flow)

            dice, jacc = mask_metrics(label1, label2_warped)

            dice = dice.cpu().numpy()[0]
            jacc = jacc.cpu().numpy()[0]
            jacb = jacobian_det(agg_flow.cpu().numpy()[0])

            if args.local_rank == 0:
                print('Pair{:6d}   dice:{:10.6f}   jacc:{:10.6f}   jacb:{:10.2f}'.
                      format(i, dice, jacc, jacb),
                      file=f)

            Dice.append(dice)
            Jacc.append(jacc)
            Jacb.append(jacb)

        dice_mean, dice_std = np.mean(np.array(Dice)), np.std(np.array(Dice))
        jacc_mean, jacc_std = np.mean(np.array(Jacc)), np.std(np.array(Jacc))
        jacb_mean, jacb_std = np.mean(np.array(Jacb)), np.std(np.array(Jacb))

        if args.local_rank == 0:
            print('Summary --->  '
                  'Dice:{:10.6f}({:10.6f})   Jacc:{:10.6f}({:10.6f})  '
                  'Jacb:{:10.2f}({:10.2f})'
                  .format(dice_mean, dice_std, jacc_mean, jacc_std, jacb_mean, jacb_std),
                  file=f)

            print('Step{:12d} --->  '
                  'Dice:{:10.6f}({:10.6f})   Jacc:{:10.6f}({:10.6f})  '
                  'Jacb:{:10.2f}({:10.2f})'
                  .format(steps, dice_mean, dice_std, jacc_mean, jacc_std, jacb_mean, jacb_std),
                  file=g)

        f.close()
        g.close()


def evaluate_brain(args, model, steps):
    for datas in args.dataset_test:
        eval_path = join(args.eval_path, datas)
        if (args.local_rank == 0) and (not os.path.isdir(eval_path)):
            os.makedirs(eval_path)
        file_sum = join(eval_path, datas + '.txt')
        file = join(eval_path, datas + '_' + str(steps) + '.txt')
        f = open(file, 'a+')
        g = open(file_sum, 'a+')

        Dice, Jacc, Jacb = [], [], []
        eval_dataset = datasets.BrainTest(args, datas)
        if args.local_rank == 0:
            print('Dataset in evaluation: %s' % datas, file=f)
            print('Image pairs in evaluation: %d' % len(eval_dataset), file=f)
            print('Evaluation steps: %s' % steps, file=f)

        for i in range(len(eval_dataset)):
            image1, image2 = eval_dataset[i][0][np.newaxis].cuda(), eval_dataset[i][1][np.newaxis].cuda()
            label1, label2 = eval_dataset[i][2][np.newaxis].cuda(), eval_dataset[i][3][np.newaxis].cuda()

            with torch.no_grad():
                _, _, _, agg_flow, _ = model.module(image1, image2, augment=False)

            jaccs = []
            dices = []

            for v in eval_dataset.seg_values:
                label1_fixed = mask_class(label1, v)
                label2_warped = warp3D()(mask_class(label2, v), agg_flow)

                class_dice, class_jacc = mask_metrics(label1_fixed, label2_warped)

                dices.append(class_dice)
                jaccs.append(class_jacc)

            jacb = jacobian_det(agg_flow.cpu().numpy()[0])

            dice = torch.mean(torch.cuda.FloatTensor(dices)).cpu().numpy()
            jacc = torch.mean(torch.cuda.FloatTensor(jaccs)).cpu().numpy()

            if args.local_rank == 0:
                print('Pair{:6d}   dice:{:10.6f}   jacc:{:10.6f}   jacb:{:10.2f}'.
                      format(i, dice, jacc, jacb),
                      file=f)

            Dice.append(dice)
            Jacc.append(jacc)
            Jacb.append(jacb)

        dice_mean, dice_std = np.mean(np.array(Dice)), np.std(np.array(Dice))
        jacc_mean, jacc_std = np.mean(np.array(Jacc)), np.std(np.array(Jacc))
        jacb_mean, jacb_std = np.mean(np.array(Jacb)), np.std(np.array(Jacb))

        if args.local_rank == 0:
            print('Summary --->  '
                  'Dice:{:10.6f}({:10.6f})   Jacc:{:10.6f}({:10.6f})  '
                  'Jacb:{:10.2f}({:10.2f})'
                  .format(dice_mean, dice_std, jacc_mean, jacc_std, jacb_mean, jacb_std),
                  file=f)

            print('Step{:12d} --->  '
                  'Dice:{:10.6f}({:10.6f})   Jacc:{:10.6f}({:10.6f})  '
                  'Jacb:{:10.2f}({:10.2f})'
                  .format(steps, dice_mean, dice_std, jacc_mean, jacc_std, jacb_mean, jacb_std),
                  file=g)

        f.close()
        g.close()


def evaluate(args, model, steps=100000):
    if args.dataset == 'liver':
        evaluate_liver(args, model, steps)
    elif args.dataset == 'brain':
        evaluate_brain(args, model, steps)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='SDHNet', help='evaluation experiment')
    parser.add_argument('--model', type=str, default='SDHNet_lpba', help='evaluation experiment')
    parser.add_argument('--dataset', type=str, default='brain', help='which dataset to use for evaluation')
    parser.add_argument('--dataset_test', nargs='+', default=['lpba'], help='specific dataset to use for evaluation')
    parser.add_argument('--data_path', type=str, default='E:/Registration/Code/TMI2022/Github/Data_MRIBrain/')
    parser.add_argument('--base_path', type=str, default='E:/Registration/Code/TMI2022/Github/')
    parser.add_argument('--iters', type=int, default=6)
    parser.add_argument('--local_rank', default=0, type=int, help='node rank for distributed training')
    args = parser.parse_args()

    args.model_path = args.base_path + args.name + '/output/checkpoints_' + args.dataset
    args.eval_path = args.base_path + args.name + '/output/eval_' + args.dataset
    args.restore_ckpt = join(args.model_path, args.model + '.pth')

    model = Framework(args)
    model = nn.DataParallel(model)
    model.eval()
    model.cuda()

    model.load_state_dict(torch.load(args.restore_ckpt))
    evaluate(args, model)
