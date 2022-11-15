import numpy as np
import torch


def det3x3(M):
    M = [[M[:, i, j] for j in range(3)] for i in range(3)]

    det = (M[0][0] * M[1][1] * M[2][2] + M[0][1] * M[1][2] * M[2][0] + M[0][2] * M[1][0] * M[2][1]) - \
          (M[0][0] * M[1][2] * M[2][1] + M[0][1] * M[1][0] * M[2][2] + M[0][2] * M[1][1] * M[2][0])

    return det


def elem_sym_polys_of_eigen_values(M):
    M = [[M[:, i, j] for j in range(3)] for i in range(3)]

    sigma1 = (M[0][0] + M[1][1] + M[2][2])

    sigma2 = (M[0][0] * M[1][1] + M[1][1] * M[2][2] + M[2][2] * M[0][0]) - \
             (M[0][1] * M[1][0] + M[1][2] * M[2][1] + M[2][0] * M[0][2])

    sigma3 = (M[0][0] * M[1][1] * M[2][2] + M[0][1] * M[1][2] * M[2][0] + M[0][2] * M[1][0] * M[2][1]) - \
             (M[0][0] * M[1][2] * M[2][1] + M[0][1] * M[1][0] * M[2][2] + M[0][2] * M[1][1] * M[2][0])

    return sigma1, sigma2, sigma3


def similarity_loss(img1, img2_warped):
    sizes = np.prod(img1.shape[1:])
    flatten1 = img1.view(-1, sizes)
    flatten2 = img2_warped.view(-1, sizes)

    mean1 = torch.mean(flatten1, -1).view(-1, 1)
    mean2 = torch.mean(flatten2, -1).view(-1, 1)
    var1 = torch.mean((flatten1 - mean1) ** 2, -1)
    var2 = torch.mean((flatten2 - mean2) ** 2, -1)

    conv12 = torch.mean((flatten1 - mean1) * (flatten2 - mean2), -1)
    pearson_r = conv12 / torch.sqrt((var1 + 1e-6) * (var2 + 1e-6))

    raw_loss = 1 - pearson_r
    raw_loss = torch.sum(raw_loss)

    return raw_loss


def regularize_loss(flow):
    ret = torch.sum((flow[:, :, 1:, :, :] - flow[:, :, :-1, :, :]) ** 2) / 2 + \
          torch.sum((flow[:, :, :, 1:, :] - flow[:, :, :, :-1, :]) ** 2) / 2 + \
          torch.sum((flow[:, :, :, :, 1:] - flow[:, :, :, :, :-1]) ** 2) / 2
    ret = ret / np.prod(flow.shape[1:5])

    return ret


def dist_value(flow1, flow2):
    ret = torch.mean((flow1 - flow2) ** 2).sqrt()
    return ret


def dist_spatialgra(flow1, flow2):
    ret1 = torch.mean(((flow1[:, :, 1:, :, :] - flow1[:, :, :-1, :, :]) -
                       (flow2[:, :, 1:, :, :] - flow2[:, :, :-1, :, :])) ** 2).sqrt()

    ret2 = torch.mean(((flow1[:, :, :, 1:, :] - flow1[:, :, :, :-1, :]) -
                       (flow2[:, :, :, 1:, :] - flow2[:, :, :, :-1, :])) ** 2).sqrt()

    ret3 = torch.mean(((flow1[:, :, :, :, 1:] - flow1[:, :, :, :, :-1]) -
                       (flow2[:, :, :, :, 1:] - flow2[:, :, :, :, :-1])) ** 2).sqrt()

    return (ret1+ret2+ret3) / 3.0


def distill_loss(AGG_flows, gamma=0.8):
    final_flow = AGG_flows[-1].detach()

    nums = len(AGG_flows)

    dist_loss = 0.0
    for i in range(1, nums):
        weight = gamma ** (nums-1-i)
        dist_loss_1 = dist_value(final_flow, AGG_flows[i-1])
        dist_loss_2 = dist_spatialgra(final_flow, AGG_flows[i-1])

        dist_loss = dist_loss + weight * (dist_loss_1 + dist_loss_2)

    return dist_loss / (nums-1)
