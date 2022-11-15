import torch
import torch.nn as nn


class AffineNet(nn.Module):
    def __init__(self, multiplier=1):
        super(AffineNet, self).__init__()
        self.multiplier = multiplier

        self.conv1 = nn.Conv3d(2,  16, kernel_size=3, stride=2, padding=1)  # 64 * 64 * 64

        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1)  # 32 * 32 * 32

        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1)  # 16 * 16 * 16
        self.conv3_1 = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv4 = nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1)  # 8 * 8 * 8
        self.conv4_1 = nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv5 = nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1)  # 4 * 4 * 4
        self.conv5_1 = nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv6 = nn.Conv3d(256, 512, kernel_size=3, stride=2, padding=1)  # 2 * 2 * 2
        self.conv6_1 = nn.Conv3d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv7_W = nn.Conv3d(512, 9, kernel_size=2, stride=1, padding=0, bias=False)
        self.conv7_b = nn.Conv3d(512, 3, kernel_size=2, stride=1, padding=0, bias=False)

        self.lrelu = nn.LeakyReLU(0.1, inplace=True)

        #  weight initialization ?

    def affine_flow(self, W, b, sd, sh, sw):
        b = b.view([-1, 3, 1, 1, 1])

        xr = torch.arange(-(sw - 1) / 2.0, sw / 2.0, 1.0)
        xr = xr.view([1, 1, 1, 1, -1]).cuda()
        yr = torch.arange(-(sh - 1) / 2.0, sh / 2.0, 1.0)
        yr = yr.view([1, 1, 1, -1, 1]).cuda()
        zr = torch.arange(-(sd - 1) / 2.0, sd / 2.0, 1.0)
        zr = zr.view([1, 1, -1, 1, 1]).cuda()

        wx = W[:, :, 0]
        wx = wx.view([-1, 3, 1, 1, 1])
        wy = W[:, :, 1]
        wy = wy.view([-1, 3, 1, 1, 1])
        wz = W[:, :, 2]
        wz = wz.view([-1, 3, 1, 1, 1])

        return xr * wx + yr * wy + zr * wz + b

    def forward(self, image1, image2):

        concatImgs = torch.cat([image1, image2], 1)  # B, C, D, H, W

        x = self.lrelu(self.conv1(concatImgs))
        x = self.lrelu(self.conv2(x))
        x = self.lrelu(self.conv3(x))
        x = self.lrelu(self.conv3_1(x))
        x = self.lrelu(self.conv4(x))
        x = self.lrelu(self.conv4_1(x))
        x = self.lrelu(self.conv5(x))
        x = self.lrelu(self.conv5_1(x))
        x = self.lrelu(self.conv6(x))
        x = self.lrelu(self.conv6_1(x))

        I = torch.cuda.FloatTensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]])
        W = self.conv7_W(x).view([-1, 3, 3]) * self.multiplier
        b = self.conv7_b(x).view([-1, 3]) * self.multiplier

        A = W + I

        sd, sh, sw = image1.shape[2:5]
        flow = self.affine_flow(W, b, sd, sh, sw)  # B, C, D, H, W (Displacement Field)

        return {'flow': flow, 'A': A, 'W': W, 'b': b}
