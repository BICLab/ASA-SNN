import torch
import torch.nn as nn
import torch.nn.functional as F

from models import POOL_LAYER, ATTN_LAYER
from models.module.TAda import conv_TAda
from spikingjelly.clock_driven.neuron import *
from spikingjelly.clock_driven import layer, functional, surrogate
# Incorporating Learnable Membrane Time Constant to Enhance Learning of Spiking Neural Networks https://arxiv.org/abs/2007.05785


class VotingLayer(nn.Module):
    def __init__(self, voting_size: int = 10):
        super().__init__()
        self.voting_size = voting_size

    def forward(self, x: torch.Tensor):
        y = F.avg_pool1d(x.unsqueeze(1), self.voting_size, self.voting_size).squeeze(1)
        return y


class ConvAttLIF(nn.Module):
    def __init__(
        self,
        attention,
        inputSize,
        hiddenSize,
        kernel_size,
        stride,
        T,
        pool_mode="no",
        track_running_stats=True,
        TAda=False,
        attention_before_spike=True
    ):
        super().__init__()

        self.attention_before_spike = attention_before_spike
        self.TAda = TAda

        self.pool = layer.SeqToANNContainer(POOL_LAYER[pool_mode])

        if not TAda:
            self.conv = layer.SeqToANNContainer(nn.Conv2d(
                in_channels=inputSize,
                out_channels=hiddenSize,
                kernel_size=kernel_size,
                padding=1,
                stride=stride,
                bias=False,
            ))
        else:
            self.conv = conv_TAda(
                in_planes=inputSize,
                out_planes=hiddenSize,
                kernel_size=kernel_size,
                stride=stride,
            )

        self.bn = layer.SeqToANNContainer(nn.BatchNorm2d(
            hiddenSize, track_running_stats=track_running_stats
        ))

        self.spike = layer.MultiStepContainer(ParametricLIFNode(
            surrogate_function=surrogate.ATan()
        ))

        self.attn = ATTN_LAYER[attention](T, hiddenSize)

    def forward(self, x):
        x = self.pool(x)

        if not self.TAda:
            x = self.conv(x)
        else:
            x = x.permute(1, 2, 0, 3, 4).contiguous()
            x = self.conv(x, x)
            x = x.permute(2, 0, 1, 3, 4).contiguous()

        x = self.bn(x)

        if self.attention_before_spike:
            x = x.transpose(0, 1).contiguous()
            x = self.attn(x)
            x = x.transpose(0, 1).contiguous()

        x = self.spike(x)

        if not self.attention_before_spike:
            x = x.transpose(0, 1).contiguous()
            x = self.attn(x)
            x = x.transpose(0, 1).contiguous()

        return x


class AttLIF(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
    ):
        super().__init__()

        self.fc = nn.Linear(in_channels, out_channels)
        self.spike = layer.MultiStepContainer(ParametricLIFNode(
            surrogate_function=surrogate.ATan()
        ))

    def forward(self, x):
        x = functional.seq_to_ann_forward(x, self.fc)
        x = self.spike(x)
        return x


class DVSGestureNet(nn.Module):
    def __init__(
        self, 
        args, 
        channels=128, 
        conv_configs = [
            (2, 128, 3, 1),
            (128, 128, 3, 1),
            (128, 128, 3, 1),
            (128, 128, 3, 1),
            (128, 128, 3, 1),
        ],
        pool_modes = ["no", "max", "max", "max", "max"],
        attn_flags = [1, 1, 1, 1, 1],
    ):
        super().__init__()
        assert len(pool_modes) == len(conv_configs)

        TAda_configs = [False]
        for _ in range(len(conv_configs)-1):
            TAda_configs.append(args.TAda)

        ConvAttLIFs = []
        for i, (conv_config, pool_mode, TAda, attn_flag) in enumerate(zip(conv_configs, pool_modes, TAda_configs, attn_flags)):
            ConvAttLIFs.append(
                nn.Sequential(
                    ConvAttLIF(
                        attention=args.attention,
                        inputSize=conv_config[0],
                        hiddenSize=conv_config[1],
                        kernel_size=conv_config[2],
                        stride=conv_config[3],
                        pool_mode=pool_mode,
                        T=args.T,
                        track_running_stats = args.track_running_stats,
                        TAda=TAda,
                    ),
                )
            )
        self.pool = layer.SeqToANNContainer(POOL_LAYER["max"])
        self.ConvAttLIFs = nn.ModuleList(ConvAttLIFs)

        FCs = []
        cfg_fc = (channels * 4 ** 2, 512, args.num_classes * 10)
        if args.dataset == "action" or args.dataset == "recogition":
            if args.ds <= 2:
                cfg_fc = (channels * 80 // (args.ds ** 2), 512, args.num_classes * 10)
            elif args.ds == 4:
                cfg_fc = (512, 512, args.num_classes * 10)

        for i in range(len(cfg_fc)-1):
            FCs.append(
                nn.Sequential(
                    layer.MultiStepDropout(0.0),
                    AttLIF(cfg_fc[i], cfg_fc[i+1])
                )
            )
        self.FCs = nn.ModuleList(FCs)

        self.voting = VotingLayer(10)

    def forward(self, x: torch.Tensor):
        x = x.transpose(0, 1).contiguous()
        firing_nums = []

        for body_layer in self.ConvAttLIFs:
            x = body_layer(x)
            firing_nums.append(x)

        x = self.pool(x)
        x = x.flatten(2)

        for head_layer in self.FCs:
            x = head_layer(x)
            firing_nums.append(x)

        x = functional.seq_to_ann_forward(x, self.voting)

        x = torch.mean(x, dim=0)

        return x, firing_nums
