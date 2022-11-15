import numpy as np
import torch
import torch.nn.functional as F


class warp3D:
    def __init__(self, padding=False):
        self.padding = padding

    def __call__(self, I, flow):
        return self._transform(I, flow[:, 0, :, :, :], flow[:, 1, :, :, :], flow[:, 2, :, :, :])

    def _meshgrid(self, depth, height, width):
        x_t = torch.matmul(torch.ones(height, 1),
                           (torch.linspace(0.0, float(width) - 1.0, width)[:, np.newaxis].permute(1, 0).contiguous()))
        x_t = x_t[np.newaxis].repeat(depth, 1, 1)

        y_t = torch.matmul(torch.linspace(0.0, float(height) - 1.0, height)[:, np.newaxis], torch.ones(1, width))
        y_t = y_t[np.newaxis].repeat(depth, 1, 1)

        z_t = torch.linspace(0.0, float(depth) - 1.0, depth)[:, np.newaxis, np.newaxis].repeat(1, height, width)

        return x_t, y_t, z_t

    def _transform(self, I, dx, dy, dz):
        batch_size = dx.shape[0]
        depth = dx.shape[1]
        height = dx.shape[2]
        width = dx.shape[3]

        # Convert dx and dy to absolute locations
        x_mesh, y_mesh, z_mesh = self._meshgrid(depth, height, width)
        x_mesh = x_mesh[np.newaxis]
        y_mesh = y_mesh[np.newaxis]
        z_mesh = z_mesh[np.newaxis]

        x_mesh = x_mesh.repeat(batch_size, 1, 1, 1).cuda()
        y_mesh = y_mesh.repeat(batch_size, 1, 1, 1).cuda()
        z_mesh = z_mesh.repeat(batch_size, 1, 1, 1).cuda()
        x_new = dx + x_mesh
        y_new = dy + y_mesh
        z_new = dz + z_mesh

        return self._interpolate(I, x_new, y_new, z_new)

    def _repeat(self, x, n_repeats):
        rep = torch.ones(size=[n_repeats, ])[:, np.newaxis].permute(1, 0).contiguous().int()
        x = torch.matmul(x.view([-1, 1]).int(), rep)
        return x.view([-1])

    def _interpolate(self, im, x, y, z):
        if self.padding:
            im = F.pad(im, (1, 1, 1, 1, 1, 1))

        num_batch = im.shape[0]
        channels = im.shape[1]
        depth = im.shape[2]
        height = im.shape[3]
        width = im.shape[4]

        out_depth = x.shape[1]
        out_height = x.shape[2]
        out_width = x.shape[3]

        x = x.view([-1])
        y = y.view([-1])
        z = z.view([-1])

        padding_constant = 1 if self.padding else 0
        x = x.float() + padding_constant
        y = y.float() + padding_constant
        z = z.float() + padding_constant

        max_x = int(width - 1)
        max_y = int(height - 1)
        max_z = int(depth - 1)

        x0 = torch.floor(x).int()
        x1 = x0 + 1
        y0 = torch.floor(y).int()
        y1 = y0 + 1
        z0 = torch.floor(z).int()
        z1 = z0 + 1

        x0 = torch.clamp(x0, 0, max_x)
        x1 = torch.clamp(x1, 0, max_x)
        y0 = torch.clamp(y0, 0, max_y)
        y1 = torch.clamp(y1, 0, max_y)
        z0 = torch.clamp(z0, 0, max_z)
        z1 = torch.clamp(z1, 0, max_z)

        dim1 = width
        dim2 = width * height
        dim3 = width * height * depth

        base = self._repeat(torch.arange(num_batch) * dim3,
                            out_depth * out_height * out_width).cuda()

        idx_a = (base + x0 + y0 * dim1 + z0 * dim2)[:, np.newaxis].repeat(1, channels)
        idx_b = (base + x0 + y1 * dim1 + z0 * dim2)[:, np.newaxis].repeat(1, channels)
        idx_c = (base + x1 + y0 * dim1 + z0 * dim2)[:, np.newaxis].repeat(1, channels)
        idx_d = (base + x1 + y1 * dim1 + z0 * dim2)[:, np.newaxis].repeat(1, channels)
        idx_e = (base + x0 + y0 * dim1 + z1 * dim2)[:, np.newaxis].repeat(1, channels)
        idx_f = (base + x0 + y1 * dim1 + z1 * dim2)[:, np.newaxis].repeat(1, channels)
        idx_g = (base + x1 + y0 * dim1 + z1 * dim2)[:, np.newaxis].repeat(1, channels)
        idx_h = (base + x1 + y1 * dim1 + z1 * dim2)[:, np.newaxis].repeat(1, channels)

        # use indices to lookup pixels in the flat image and restore
        # channels dim
        im_flat = im.permute(0, 2, 3, 4, 1).contiguous().view([-1, channels]).float()

        Ia = torch.gather(im_flat, 0, idx_a.long())
        Ib = torch.gather(im_flat, 0, idx_b.long())
        Ic = torch.gather(im_flat, 0, idx_c.long())
        Id = torch.gather(im_flat, 0, idx_d.long())
        Ie = torch.gather(im_flat, 0, idx_e.long())
        If = torch.gather(im_flat, 0, idx_f.long())
        Ig = torch.gather(im_flat, 0, idx_g.long())
        Ih = torch.gather(im_flat, 0, idx_h.long())

        # and finally calculate interpolated values
        x1_f = x1.float()
        y1_f = y1.float()
        z1_f = z1.float()

        dx = x1_f - x
        dy = y1_f - y
        dz = z1_f - z

        wa = (dz * dx * dy)[:, np.newaxis]
        wb = (dz * dx * (1 - dy))[:, np.newaxis]
        wc = (dz * (1 - dx) * dy)[:, np.newaxis]
        wd = (dz * (1 - dx) * (1 - dy))[:, np.newaxis]
        we = ((1 - dz) * dx * dy)[:, np.newaxis]
        wf = ((1 - dz) * dx * (1 - dy))[:, np.newaxis]
        wg = ((1 - dz) * (1 - dx) * dy)[:, np.newaxis]
        wh = ((1 - dz) * (1 - dx) * (1 - dy))[:, np.newaxis]

        output = wa * Ia + wb * Ib + wc * Ic + wd * Id + we * Ie + wf * If + wg * Ig + wh * Ih
        output = output.view([-1, out_depth, out_height, out_width, channels])

        return output.permute(0, 4, 1, 2, 3)
