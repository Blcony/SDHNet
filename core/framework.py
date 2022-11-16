import os
import torch.distributed as dist
import random
import numpy as np
import torch
import torch.nn as nn
from .utils import aug_transform, warp
from .networks import affnet, sdhnet


def init(args):
    if not os.path.isdir(args.model_path):
        os.makedirs(args.model_path)
    if not os.path.isdir(args.eval_path):
        os.makedirs(args.eval_path)
    dist.init_process_group(backend='nccl')
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)


class Framework(nn.Module):
    def __init__(self, args):
        super(Framework, self).__init__()
        self.args = args
        self.cdim = 32
        self.hdim = 16
        self.flow_multiplier = 1.0 / args.iters
        self.sample_power = aug_transform.sample_power
        self.free_form_fields = aug_transform.free_form_fields
        self.reconstruction = warp.warp3D()

        self.affnet = affnet.AffineNet()
        self.context = sdhnet.ContextNet(outputc=self.cdim + self.hdim)
        self.defnet = nn.ModuleList([sdhnet.SDHNet(hdim=self.hdim, flow_multiplier=self.flow_multiplier) for _ in range(args.iters)])

    def forward(self, Img1, Img2, augment=True):
        if augment:
            bs = Img1.shape[0]
            imgs = Img1.shape[2:5]  # D, H, W

            control_fields = (self.sample_power(-0.4, 0.4, 3, [bs, 5, 5, 5, 3]) *
                              torch.Tensor(np.array(imgs).astype(np.float) // 4)).permute(0, 4, 1, 2, 3)
            augFlow = (self.free_form_fields(imgs, control_fields)).cuda()  # B, C, D, H, W

            augImg2 = self.reconstruction(Img2, augFlow)  # B, C, D, H, W
        else:
            augImg2 = Img2

        affines = self.affnet(Img1, augImg2)

        contexts = self.context(Img1)  # extract the features of the fixed image
        cont, hid = torch.split(contexts, [self.cdim, self.hdim], dim=1)
        cont = torch.relu(cont)
        hid = [torch.tanh(hid),
               torch.tanh(torch.max_pool3d(hid, kernel_size=2, stride=2)),
               torch.tanh(torch.max_pool3d(hid, kernel_size=4, stride=4))]

        augImg2_affine = self.reconstruction(augImg2, affines['flow'])
        deforms_0, hid = self.defnet[0](Img1, augImg2_affine, cont, hid)

        I = torch.cuda.FloatTensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]])
        agg_flow_0 = torch.einsum('bij,bjxyz->bixyz', affines['W'] + I, deforms_0['flow']) + affines['flow']

        warpImg = self.reconstruction(augImg2, agg_flow_0)
        agg_flow = agg_flow_0

        Deforms = [deforms_0]
        AGG_flows = []
        for i in range(self.args.iters - 1):
            deforms, hid = self.defnet[i + 1](Img1, warpImg, cont, hid)
            agg_flow = self.reconstruction(agg_flow, deforms['flow']) + deforms['flow']
            warpImg = self.reconstruction(augImg2, agg_flow)

            Deforms.append(deforms)
            AGG_flows.append(agg_flow)

        return augImg2, affines, Deforms, agg_flow, AGG_flows
