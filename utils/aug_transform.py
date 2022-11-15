import numpy as np
import torch
import torch.nn.functional as F


def get_coef(u):
    return torch.stack([((1 - u) ** 3) / 6, (3 * (u ** 3) - 6 * (u ** 2) + 4) / 6,
                        (-3 * (u ** 3) + 3 * (u ** 2) + 3 * u + 1) / 6, (u ** 3) / 6], dim=1)


def sample_power(lo, hi, k, size=None):
    r = (hi - lo) / 2
    center = (hi + lo) / 2
    r = r ** (1 / k)
    points = (torch.rand(size) - 0.5) * 2 * r
    points = (torch.abs(points) ** k) * torch.sign(points)
    return points + center


def pad_3d(mat, pad):
    return F.pad(mat, (pad, pad, pad, pad, pad, pad))


def free_form_fields(shape, control_fields, padding='same'):
    interpolate_range = 4

    control_fields = torch.Tensor(control_fields)
    _, _, n, m, t = control_fields.shape
    if padding == 'same':
        control_fields = pad_3d(control_fields, 1)
    elif padding == 'valid':
        n -= 2
        m -= 2
        t -= 2

    control_fields = torch.reshape(control_fields.permute(2, 3, 4, 0, 1).contiguous(), [n + 2, m + 2, t + 2, -1])

    assert shape[0] % (n - 1) == 0
    s_x = shape[0] // (n - 1)
    u_x = (torch.arange(0, s_x, dtype=torch.float32) + 0.5) / s_x  # s_x
    coef_x = get_coef(u_x)  # (s_x, 4)

    shape_cf = control_fields.shape
    flow = torch.cat([torch.matmul(coef_x,
                                   torch.reshape(control_fields[i: i + interpolate_range], [interpolate_range, -1]))
                      for i in range(0, n - 1)], dim=0)

    assert shape[1] % (m - 1) == 0
    s_y = shape[1] // (m - 1)
    u_y = (torch.arange(0, s_y, dtype=torch.float32) + 0.5) / s_y  # s_y
    coef_y = get_coef(u_y)  # (s_y, 4)

    flow_dims = np.arange(0, len(flow.shape))[::-1]
    flow = torch.reshape(flow.permute(*flow_dims).contiguous(), [shape_cf[1], -1])
    flow = torch.cat([torch.matmul(coef_y,
                                   torch.reshape(flow[i: i + interpolate_range], [interpolate_range, -1]))
                      for i in range(0, m - 1)], dim=0)

    assert shape[2] % (t - 1) == 0
    s_z = shape[2] // (t - 1)
    u_z = (torch.arange(0, s_z, dtype=torch.float32) + 0.5) / s_z  # s_y
    coef_z = get_coef(u_z)  # (s_y, 4)

    flow_dims = np.arange(0, len(flow.shape))[::-1]
    flow = torch.reshape(flow.permute(*flow_dims).contiguous(), [shape_cf[2], -1])
    flow = torch.cat([torch.matmul(coef_z,
                                   torch.reshape(flow[i: i + interpolate_range], [interpolate_range, -1]))
                      for i in range(0, t - 1)], dim=0)

    flow = torch.reshape(flow, [shape[2], -1, 3, shape[1], shape[0]])
    flow = flow.permute(1, 2, 4, 3, 0).contiguous()

    return flow
