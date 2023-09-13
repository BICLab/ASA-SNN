import torch
import numpy as np
import torch.nn as nn
from spikingjelly.clock_driven import layer

from models import ATTN_LAYER
from utils import paramInit


class IFCell(nn.Module):
    def __init__(
        self,
        inputSize,
        hiddenSize,
        spikeActFun,
        pa_dict=None,
        pa_train_self=False,
        p=0,
        mode_select='spike',
        mem_act=torch.relu,
        TR_model='NTR',
        attention_flag='no',
        has_conv=False,
        attention_in_spike=False,
    ):
        super().__init__()
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.spikeActFun = spikeActFun

        self.UseDropOut = True
        self.batchSize = None
        self.pa_dict = pa_dict
        self.pa_train_self = pa_train_self
        self.p = p
        self.has_conv = has_conv

        self.attention_flag = attention_flag
        self.attention_in_spike = attention_in_spike
        self.attn = ATTN_LAYER[attention_flag](1, hiddenSize, reduction=1)

        # LIAF
        self.TR_model = TR_model
        # LIF
        self.mode_select = mode_select
        self.mem_act = mem_act

        if not pa_train_self and pa_dict is None:
            pa_dict = {'alpha': 0.3, 'beta': 0., 'Vreset': 0., 'Vthres': 0.6}
        self.pa_dict = pa_dict

        if self.pa_train_self:
            self.alpha = nn.Parameter(torch.Tensor(1, hiddenSize))
            self.beta = nn.Parameter(torch.Tensor(1, hiddenSize))
            self.Vreset = nn.Parameter(torch.Tensor(1, hiddenSize))
            self.Vthres = nn.Parameter(torch.Tensor(1, hiddenSize))
            nn.init.uniform_(self.alpha, 0, 1)
            nn.init.uniform_(self.beta, 0, 1)
            nn.init.uniform_(self.Vreset, 0, 1)
            nn.init.uniform_(self.Vthres, 0, 1)
        else:
            self.alpha = self.pa_dict['alpha']
            self.beta = self.pa_dict['beta']
            self.Vreset = self.pa_dict['Vreset']
            self.Vthres = self.pa_dict['Vthres']

        if 0 < self.p < 1:
            self.dropout = layer.Dropout(p=self.p)
    
    def _forward_impl(self, input):
        input = input.reshape([self.batchSize, -1])
        return input
    
    def _forward_impl_attn(self, input):
        input = self.attn(input.unsqueeze(1))
        input = input.squeeze(1)
        return input

    def forward(self, input, init_v=None):
        self.batchSize = input.size()[0]

        if not self.has_conv:
            input = self._forward_impl(input)

        if not hasattr(self, "h"):
            if init_v is None:
                if self.has_conv:
                    self.h = torch.zeros(
                        self.batchSize,
                        self.hiddenSize,
                        input.size()[-2],
                        input.size()[-1],
                        device=input.device
                    )
                else:
                    self.h = torch.zeros(
                        self.batchSize,
                        self.hiddenSize,
                        device=input.device
                    )
            else:
                self.h = init_v.clone()

        if input.device != self.h.device:
            input = input.to(self.h.device)

        # Step 1: accumulate and reset,spike used as forgetting gate
        u = self.h + input

        if self.has_conv and self.attention_in_spike:
            u = self._forward_impl_attn(u)

        x_ = u - self.Vthres
        x = self.spikeActFun(x_)
        
        self.h = x * self.Vreset + (1 - x) * u
        # self.h = x * self.Vthres + (1 - x) * u
        self.h = self.h * self.alpha + self.beta

        # step 4:
        if self.mode_select == 'spike':
            x = x
        elif self.mode_select == 'mem':
            # TR
            if self.TR_model == 'TR':
                if not self.mem_act:
                    x = x_
                else:
                    x = self.mem_act(x_)
            else:
                if not self.mem_act:
                    x = u
                else:
                    x = self.mem_act(u)

        if 1 > self.p > 0:
            x = self.dropout(x)
        return x

    def reset(self):
        if hasattr(self, "h"):
            del self.h


class AttLIF(nn.Module):
    def __init__(
        self,
        inputSize,
        hiddenSize,
        spikeActFun,
        attention='T',
        onlyLast=False,
        useBatchNorm=False,
        init_method=None,
        pa_dict=None,
        pa_train_self=False,
        bias=True,
        reduction=1,
        T=1,
        p=0,
        track_running_stats=False,
        mode_select='spike',
        mem_act=torch.relu,
        TR_model='NTR',
    ):
        super().__init__()
        self.onlyLast = onlyLast
        self.useBatchNorm = useBatchNorm
        self.network = nn.Sequential()

        self.attention_flag = attention
        self.linear = layer.SeqToANNContainer(nn.Linear(
            in_features=inputSize,
            out_features=hiddenSize,
            bias=bias,
        ))
        if init_method is not None:
            paramInit(model=self.linear, method=init_method)

        if self.useBatchNorm:
            self.bn = layer.SeqToANNContainer(nn.BatchNorm1d(
                num_features=hiddenSize,
                track_running_stats=track_running_stats
            ))

        assert reduction <= hiddenSize, \
            "the attn_channel should be bigger than the reduction"
        self.attn = ATTN_LAYER[attention](T, hiddenSize, reduction=reduction)

        self.dropout = layer.Dropout(p)

        self.spike = layer.MultiStepContainer(IFCell(
            inputSize,
            hiddenSize,
            spikeActFun,
            pa_dict=pa_dict,
            pa_train_self=pa_train_self,
            p=p,
            mode_select=mode_select,
            mem_act=mem_act,
            TR_model=TR_model,
        ))

    def forward(self, data, mp_collect=False):
        data = self.dropout(data)
        output = self.linear(data)
        if self.useBatchNorm:
            output = self.bn(output)
        if self.onlyLast:
            return output

        mp = output
        mp = self.attn(mp.permute(1, 0, 2).contiguous()).permute(1, 0, 2).contiguous()
        output = self.spike(mp)
        # print(np.sum(mp.cpu().detach().numpy()>0.3))
        # print(np.sum(output.cpu().detach().numpy()>0.3))
        # print()

        if mp_collect:
            return output, mp
        else:
            return output


class ConvAttLIF(nn.Module):
    def __init__(
        self,
        inputSize,
        hiddenSize,
        kernel_size,
        spikeActFun,
        attention='no',
        onlyLast=False,
        padding=1,
        useBatchNorm=False,
        init_method=None,
        pa_dict=None,
        pa_train_self=False,
        reduction=1,
        T=1,
        stride=1,
        pooling_kernel_size=1,
        p=0,
        track_running_stats=False,
        mode_select='spike',
        mem_act=torch.relu,
        TR_model='NTR',
        attention_before_conv=False,
        attention_per_time=False,
        attention_in_spike=False,
    ):
        super().__init__()

        self.onlyLast = onlyLast
        self.attention_flag = attention
        self.attention_before_conv = attention_before_conv
        self.attention_per_time = attention_per_time

        self.conv = layer.SeqToANNContainer(nn.Conv2d(
            in_channels=inputSize,
            out_channels=hiddenSize,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
        ))

        if init_method is not None:
            paramInit(model=self.conv, method=init_method)

        self.useBatchNorm = useBatchNorm

        if self.useBatchNorm:
            self.bn = layer.SeqToANNContainer(nn.BatchNorm2d(
                hiddenSize, track_running_stats=track_running_stats
            ))

        self.pooling_kernel_size = pooling_kernel_size
        if self.pooling_kernel_size > 1:
            self.pooling = layer.SeqToANNContainer(
                nn.AvgPool2d(kernel_size=pooling_kernel_size)
            )

        if attention_before_conv:
            # assert attention != "HCSA"
            attn_channels = inputSize
        else:
            attn_channels = hiddenSize

        if attention_per_time:
            assert "T" not in attention
            T = 1

        assert reduction <= attn_channels, \
            "the attn_channel should be bigger than the reduction"
        self.attn = ATTN_LAYER[attention](T, attn_channels, reduction, 5)

        self.dropout = layer.Dropout(p)

        self.spike = layer.MultiStepContainer(IFCell(
            inputSize=inputSize,
            hiddenSize=hiddenSize,
            spikeActFun=spikeActFun,
            pa_dict=pa_dict,
            pa_train_self=pa_train_self,
            p=p,
            mode_select=mode_select,
            mem_act=mem_act,
            TR_model=TR_model,
            attention_flag=self.attention_flag,
            has_conv=True,
            attention_in_spike=attention_in_spike,
        ))

    def _forward_impl_attn(self, data):
        data = data.permute(1, 0, 2, 3, 4).contiguous()

        if self.attention_per_time:
            for step in range(data.size(1)):
                out = data[:, step, :, :, :]
                out = self.attn(out.unsqueeze(1))
                output = out.squeeze(1)

                if step == 0:
                    temp = list(output.size())
                    temp.insert(1, list(data.size())[1])
                    outputsum = torch.zeros(temp)
                    if outputsum.device != data.device:
                        outputsum = outputsum.to(data.device)

                outputsum[:, step, :, :, :] = output
        else:
            outputsum = self.attn(data)

        data = outputsum.permute(1, 0, 2, 3, 4).contiguous()
        return data

    def forward(self, data, mp_collect=False):
        data = self.dropout(data)
        if self.attention_before_conv:
            data = self._forward_impl_attn(data)

        output = self.conv(data)
        if self.useBatchNorm:
            output = self.bn(output)

        if self.pooling_kernel_size > 1:
            output = self.pooling(output)

        if self.onlyLast:
            return output

        mp = output
        if not self.attention_before_conv:
            mp = self._forward_impl_attn(mp)
        output = self.spike(mp)

        if mp_collect:
            return output, mp
        else:
            return output
