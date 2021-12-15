import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb


class CNNROIMapEncoder(nn.Module):
    def __init__(
        self,
        map_channels,
        hidden_channels,
        ROI_outdim,
        output_size,
        kernel_size,
        strides,
        input_size,
    ):
        super(CNNROIMapEncoder, self).__init__()
        self.convs = nn.ModuleList()
        self.num_channel_last = hidden_channels[-1]
        self.ROI_outdim = ROI_outdim
        x_dummy = torch.ones([map_channels, *input_size]).unsqueeze(0) * torch.tensor(
            float("nan")
        )

        for i, hidden_size in enumerate(hidden_channels):
            self.convs.append(
                nn.Conv2d(
                    map_channels if i == 0 else hidden_channels[i - 1],
                    hidden_channels[i],
                    kernel_size[i],
                    stride=strides[i],
                    padding=int((kernel_size[i] - 1) / 2),
                )
            )
            x_dummy = self.convs[i](x_dummy)

        self.fc = nn.Linear(
            ROI_outdim * ROI_outdim * self.num_channel_last, output_size
        )

    def forward(self, x, ROI):
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.2)

        x = ROI_align(x, ROI, self.ROI_outdim)
        out = [None] * len(x)
        for i in range(len(x)):
            out[i] = self.fc(x[i].flatten(start_dim=-3))

        return out


# def bilinear_interpolate(img, x, y, floattype=torch.float):
#     """Return bilinear interpolation of 4 nearest pts w.r.t to x,y from img
#     Args:
#         img (torch.Tensor): Tensor of size cxwxh. Usually one channel of feature layer
#         x (torch.Tensor): Float dtype, x axis location for sampling
#         y (torch.Tensor): Float dtype, y axis location for sampling
#     batched version

#     Returns:
#         torch.Tensor: interpolated value
#     """
#     bs = img.size(0)
#     x0 = torch.floor(x).type(torch.cuda.LongTensor)
#     x1 = x0 + 1

#     y0 = torch.floor(y).type(torch.cuda.LongTensor)
#     y1 = y0 + 1

#     x0 = torch.clamp(x0, 0, img.shape[-2] - 1)
#     x1 = torch.clamp(x1, 0, img.shape[-2] - 1)
#     y0 = torch.clamp(y0, 0, img.shape[-1] - 1)
#     y1 = torch.clamp(y1, 0, img.shape[-1] - 1)

#     Ia = [None] * bs
#     Ib = [None] * bs
#     Ic = [None] * bs
#     Id = [None] * bs
#     for i in range(bs):
#         Ia[i] = img[i, ..., y0[i], x0[i]]
#         Ib[i] = img[i, ..., y1[i], x0[i]]
#         Ic[i] = img[i, ..., y0[i], x1[i]]
#         Id[i] = img[i, ..., y1[i], x1[i]]

#     Ia = torch.stack(Ia, dim=0)
#     Ib = torch.stack(Ib, dim=0)
#     Ic = torch.stack(Ic, dim=0)
#     Id = torch.stack(Id, dim=0)

#     step = (x1.type(floattype) - x0.type(floattype)) * (
#         y1.type(floattype) - y0.type(floattype)
#     )
#     step = torch.clamp(step, 1e-3, 2)
#     norm_const = 1 / step

#     wa = (x1.type(floattype) - x) * (y1.type(floattype) - y) * norm_const
#     wb = (x1.type(floattype) - x) * (y - y0.type(floattype)) * norm_const
#     wc = (x - x0.type(floattype)) * (y1.type(floattype) - y) * norm_const
#     wd = (x - x0.type(floattype)) * (y - y0.type(floattype)) * norm_const
#     return (
#         Ia * wa.unsqueeze(1)
#         + Ib * wb.unsqueeze(1)
#         + Ic * wc.unsqueeze(1)
#         + Id * wd.unsqueeze(1)
#     )
def bilinear_interpolate(img, x, y, floattype=torch.float):
    """Return bilinear interpolation of 4 nearest pts w.r.t to x,y from img
    Args:
        img (torch.Tensor): Tensor of size cxwxh. Usually one channel of feature layer
        x (torch.Tensor): Float dtype, x axis location for sampling
        y (torch.Tensor): Float dtype, y axis location for sampling

    Returns:
        torch.Tensor: interpolated value
    """
    x0 = torch.floor(x).type(torch.cuda.LongTensor)
    x1 = x0 + 1

    y0 = torch.floor(y).type(torch.cuda.LongTensor)
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, img.shape[-2] - 1)
    x1 = torch.clamp(x1, 0, img.shape[-2] - 1)
    y0 = torch.clamp(y0, 0, img.shape[-1] - 1)
    y1 = torch.clamp(y1, 0, img.shape[-1] - 1)

    Ia = img[..., y0, x0]
    Ib = img[..., y1, x0]
    Ic = img[..., y0, x1]
    Id = img[..., y1, x1]

    step = (x1.type(floattype) - x0.type(floattype)) * (
        y1.type(floattype) - y0.type(floattype)
    )
    step = torch.clamp(step, 1e-3, 2)
    norm_const = 1 / step

    wa = (x1.type(floattype) - x) * (y1.type(floattype) - y) * norm_const
    wb = (x1.type(floattype) - x) * (y - y0.type(floattype)) * norm_const
    wc = (x - x0.type(floattype)) * (y1.type(floattype) - y) * norm_const
    wd = (x - x0.type(floattype)) * (y - y0.type(floattype)) * norm_const
    return (
        Ia * wa.unsqueeze(0)
        + Ib * wb.unsqueeze(0)
        + Ic * wc.unsqueeze(0)
        + Id * wd.unsqueeze(0)
    )


def ROI_align(features, ROI, outdim):
    """Given feature layers and scaled proposals return bilinear interpolated
    points in feature layer

    Args:
        features (torch.Tensor): Tensor of shape channels x width x height
        scaled_proposal (list of torch.Tensor): x0,y0,W1,W2,H1,H2,psi
    """

    bs, num_channels, h, w = features.shape

    xg = (
        torch.cat(
            (
                torch.arange(0, outdim).view(-1, 1) - (outdim - 1) / 2,
                torch.zeros([outdim, 1]),
            ),
            dim=-1,
        )
        / outdim
    )
    yg = (
        torch.cat(
            (
                torch.zeros([outdim, 1]),
                torch.arange(0, outdim).view(-1, 1) - (outdim - 1) / 2,
            ),
            dim=-1,
        )
        / outdim
    )
    gg = xg.view(1, -1, 2) + yg.view(-1, 1, 2)
    gg = gg.to(features.device)
    res = list()
    for i in range(bs):
        if ROI[i] is not None:
            W1 = ROI[i][..., 2:3]
            W2 = ROI[i][..., 3:4]
            H1 = ROI[i][..., 4:5]
            H2 = ROI[i][..., 5:6]
            psi = ROI[i][..., 6:]
            WH = torch.cat((W1 + W2, H1 + H2), dim=-1)
            offset = torch.cat(((W1 - W2) / 2, (H1 - H2) / 2), dim=-1)
            s = torch.sin(psi).unsqueeze(-1)
            c = torch.cos(psi).unsqueeze(-1)
            rotM = torch.cat(
                (torch.cat((c, -s), dim=-1), torch.cat((s, c), dim=-1)), dim=-2
            )
            ggi = gg * WH[..., None, None, :] - offset[..., None, None, :]
            ggi = ggi @ rotM[..., None, :, :] + ROI[i][..., None, None, 0:2]

            x_sample = ggi[..., 0].flatten()
            y_sample = ggi[..., 1].flatten()
            res.append(
                bilinear_interpolate(features[i], x_sample, y_sample).view(
                    ggi.shape[0], num_channels, *ggi.shape[1:-1]
                )
            )
        else:
            res.append(None)

    return res


if __name__ == "__main__":
    import numpy as np
    from torchvision.ops.roi_align import RoIAlign
    import pdb

    device = torch.device("cuda")
    torch.set_default_tensor_type(torch.cuda.DoubleTensor)

    # create feature layer, proposals and targets
    num_proposals = 10

    bs = 1
    features = torch.randn(bs, 10, 32, 32)

    xy = torch.rand((bs, 5, 2)) * torch.tensor([32, 32])
    WH = torch.ones((bs, 5, 1)) * torch.tensor([1, 1, 1, 1]).view(1, 1, -1)
    psi = torch.zeros(bs, 5, 1)
    ROI = torch.cat((xy, WH, psi), dim=-1)
    ROI = [ROI[i] for i in range(ROI.shape[0])]
    res1 = ROI_align(features, ROI, 6)[0].transpose(0, 1)
    pdb.set_trace()

    ROI_star = torch.cat((xy - WH[..., [0, 2]], xy + WH[..., [1, 3]]), dim=-1)[0]

    roi_align_obj = RoIAlign(6, 1, sampling_ratio=2, aligned=False)
    res2 = roi_align_obj(features, [ROI_star])

    res1 - res2
