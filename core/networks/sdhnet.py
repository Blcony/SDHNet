import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils import warp


class ContextNet(nn.Module):
    def __init__(self, outputc):
        super(ContextNet, self).__init__()

        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv3d(16, 16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv3d(32, outputc, kernel_size=3, stride=1, padding=1)

        self.lrelu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        x = self.lrelu(self.conv1(x))
        x = self.lrelu(self.conv2(x))
        x = self.lrelu(self.conv3(x))
        x = self.conv4(x)

        return x


class ExtractNet(nn.Module):
    def __init__(self):
        super(ExtractNet, self).__init__()

        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv3d(32, 48, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv3d(48, 64, kernel_size=3, stride=2, padding=1)

        self.lrelu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        x_2x = self.lrelu(self.conv1(x))
        x_4x = self.lrelu(self.conv2(x_2x))
        x_8x = self.lrelu(self.conv3(x_4x))
        x_16x = self.lrelu(self.conv4(x_8x))

        return {'1/4': x_4x, '1/8': x_8x, '1/16': x_16x}


class ConvGRU(nn.Module):
    def __init__(self, inputc, hidden_dim):
        super(ConvGRU, self).__init__()
        self.convz1 = nn.Conv3d(inputc, hidden_dim, 3, padding=1)
        self.convr1 = nn.Conv3d(inputc, hidden_dim, 3, padding=1)
        self.convq1 = nn.Conv3d(inputc, hidden_dim, 3, padding=1)

        self.conv1 = nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(hidden_dim, 3, kernel_size=3, stride=1, padding=1)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, h, x):
        # 1st round
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1)))

        h = (1-z) * h + z * q

        # flow estimation
        flow = self.conv2(self.lrelu(self.conv1(h)))

        return flow, h


class Fusion(nn.Module):
    def __init__(self, inputc1, inputc2):
        super(Fusion, self).__init__()

        self.conv1 = nn.Conv3d((inputc1 + inputc2), 48, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(48, 16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv3d(16, inputc2, kernel_size=3, stride=1, padding=1)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, cont, flows):
        x = self.lrelu(self.conv1(torch.cat([cont, flows], 1)))
        x = self.lrelu(self.conv2(x))
        k = self.conv3(x)

        b, _, d, h, w = flows.shape
        flow = (flows.view(b, -1, 3, d, h, w) * F.softmax(k.view(b, -1, 3, d, h, w), 1)).sum(1)
        return flow


class SDHNet(nn.Module):
    def __init__(self, hdim, flow_multiplier=1.):
        super(SDHNet, self).__init__()
        self.hdim = hdim
        self.flow_multiplier = flow_multiplier

        self.extraction = ExtractNet()

        self.estimator_4x = ConvGRU(inputc=64 + self.hdim, hidden_dim=self.hdim)
        self.estimator_8x = ConvGRU(inputc=96 + self.hdim, hidden_dim=self.hdim)
        self.estimator_16x = ConvGRU(inputc=128 + self.hdim, hidden_dim=self.hdim)

        self.fusion = Fusion(inputc1=32, inputc2=3*3)

        self.reconstruction = warp.warp3D()

    def forward(self, image1, image2, c_fea, h_fea):
        f_fea = self.extraction(image1)
        m_fea = self.extraction(image2)
        hid_4x, hid_8x, hid_16x = h_fea

        b, c, d_4x, h_4x, w_4x = f_fea['1/4'].shape

        # estimate the multi-resolution flow
        flow_4x, hid_4x = self.estimator_4x(hid_4x, torch.cat([f_fea['1/4'], m_fea['1/4']], 1))
        flow_8x, hid_8x = self.estimator_8x(hid_8x, torch.cat([f_fea['1/8'], m_fea['1/8']], 1))
        flow_16x, hid_16x = self.estimator_16x(hid_16x, torch.cat([f_fea['1/16'], m_fea['1/16']], 1))

        hid = [hid_4x, hid_8x, hid_16x]

        # flow fusion
        flow = self.fusion(c_fea, torch.cat([flow_4x,
                                             F.interpolate(flow_8x, (d_4x, h_4x, w_4x), mode='trilinear') * 2.0,
                                             F.interpolate(flow_16x, (d_4x, h_4x, w_4x), mode='trilinear') * 4.0], 1))

        b, c, d, h, w = image1.shape
        final_flow = F.interpolate(flow, size=(d, h, w), mode='trilinear') * 4.0

        return {'flow': final_flow * self.flow_multiplier}, hid
