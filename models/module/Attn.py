import torch
from torch import nn
from einops import rearrange


class ASA(nn.Module):
    def __init__(self, timeWindows, channels, reduction=1, dimension=5):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.max_pool = nn.AdaptiveMaxPool3d((None, 1, 1))
        self.register_parameter('alpha', nn.Parameter(torch.FloatTensor([0.5])))
        self.register_parameter('beta', nn.Parameter(torch.FloatTensor([0.5])))
        self.lam_ = 0.5
        if reduction == 1:
            self.fc = nn.Linear(timeWindows, timeWindows, bias=False)
        else:
            self.fc = nn.Sequential(
                nn.Linear(timeWindows, timeWindows // reduction, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(timeWindows // reduction, timeWindows, bias=False)
            )

        # ---------------------
        # SplitSpatialAttention
        # ---------------------
        self.conv1 = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)
        self.active = nn.Sigmoid()

    def get_im_subim_channels(self, channels_im, M):
        _, topk = torch.topk(M, dim=2, k=channels_im)
        topk_ = topk.squeeze(-1).squeeze(-1)

        important_channels = torch.zeros_like(M.squeeze(-1).squeeze(-1)).to(M.device)

        for i in range(M.shape[1]):
            important_channels[:, i, topk_[:, i]] = 1

        important_channels = important_channels.unsqueeze(-1).unsqueeze(-1)
        return important_channels

    def forward(self, x):
        n, t, c, _, _ = x.shape
        tca_avg_map = self.avg_pool(x)
        tca_max_map = self.max_pool(x)
        map_add = 0.5 * (tca_avg_map + tca_max_map) + self.alpha * tca_avg_map + self.beta * tca_max_map
        map_add = self.fc(map_add.squeeze().transpose(1, 2).contiguous()).transpose(1, 2).contiguous()

        # ---------------------
        # SplitSpatialAttention
        # ---------------------
        # map_add shape: N, T, C, 1, 1
        important_channels = self.get_im_subim_channels(int(c * self.lam_), map_add)
        important_times = self.get_im_subim_channels(int(t * self.lam_), map_add.transpose(1, 2).contiguous())
        important_times = important_times.transpose(1, 2).contiguous()

        important_tc = (important_channels + important_times) / 2
        important_tc = torch.where(important_tc == 0.5, 1, 0)
        subimportant_tc = 1. - important_tc

        important_features = important_tc * x
        subimportant_features = subimportant_tc * x

        important_features = rearrange(important_features, 'n t c h w -> n (t c) h w')
        subimportant_features = rearrange(subimportant_features, 'n t c h w -> n (t c) h w')

        im_AvgPool = torch.mean(important_features, dim=1, keepdim=True) / self.lam_
        im_MaxPool, _ = torch.max(important_features, dim=1, keepdim=True)

        subim_AvgPool = torch.mean(subimportant_features, dim=1, keepdim=True) / (1 - self.lam_)
        subim_MaxPool, _ = torch.max(subimportant_features, dim=1, keepdim=True)

        im_x = torch.cat([im_AvgPool, im_MaxPool], dim=1)
        subim_x = torch.cat([subim_AvgPool, subim_MaxPool], dim=1)

        im_map = self.active(self.conv1(im_x))
        subim_map = self.active(self.conv2(subim_x))

        important_features = im_map * important_features
        subimportant_features = subim_map * subimportant_features

        htsa_out = important_features + subimportant_features
        htsa_out = rearrange(htsa_out, 'n (t c) h w -> n t c h w', t=t)

        return htsa_out
