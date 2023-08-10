import torch
import torch.nn as nn

from models.module.LIF import ConvAttLIF, AttLIF


class DVSGestureNet(nn.Module):
    def __init__(
        self, 
        args,
        conv_configs = [
            (2, 32, 3,),
            (32, 32, 3,),
            (32, 32, 3,),
        ],
        pool_kernels_size = [1, 2, 2],
        attn_flags = [1, 1, 1],
    ):
        super().__init__()

        TAda_configs = [False]
        for _ in range(len(conv_configs)-1):
            TAda_configs.append(args.TAda)
        dropout_ps = []
        for i in range(len(args.ps)):
            if args.ps[i] == "1":
                dropout_ps.append(0.5)
            elif args.ps[i] == "0":
                dropout_ps.append(0.)

        class ActFun(torch.autograd.Function):
            def __init__(self):
                super(ActFun, self).__init__()

            @staticmethod
            def forward(ctx, input):
                ctx.save_for_backward(input)
                return input.ge(0.).float()

            @staticmethod
            def backward(ctx, grad_output):
                input, = ctx.saved_tensors
                temp = abs(input) < args.lens
                return grad_output * temp.float() / (2 * args.lens)

        ConvAttLIFs = []
        for i, (
            conv_config, 
            pooling_kernel_size, 
            TAda,
            attn_flag
        ) in enumerate(zip(
            conv_configs, 
            pool_kernels_size, 
            TAda_configs,
            attn_flags,
        )):
            if attn_flag == 0:
                attn = "no"
            elif attn_flag == 1:
                attn = args.attention
            ConvAttLIFs.append(
                ConvAttLIF(
                    p=dropout_ps[i],
                    attention=attn,
                    inputSize=conv_config[0],
                    hiddenSize=conv_config[1],
                    kernel_size=conv_config[2],
                    spikeActFun=ActFun.apply,
                    init_method=args.init_method,
                    useBatchNorm=True,
                    pooling_kernel_size=pooling_kernel_size,
                    T=args.T,
                    pa_dict={'alpha': args.alpha, 'beta': args.beta, 'Vreset': args.Vreset, 'Vthres': args.Vthres},
                    reduction= args.reduction,
                    track_running_stats = args.track_running_stats,
                    mode_select=args.mode_select,
                    mem_act=args.mem_act,
                    TR_model=args.TR_model,
                    TAda=TAda,
                    attention_before_conv=args.attention_before_conv,
                    attention_per_time=args.attention_per_time,
                    attention_in_spike=args.attention_in_spike,
                )
            )
        self.ConvAttLIFs = nn.ModuleList(ConvAttLIFs)
        for l in self.ConvAttLIFs:
            l.attn.lam_ = args.lam

        # Building Head
        FCs = []
        cfg_fc = [conv_configs[-1][1] * 8 * 8, 256, args.num_classes]
        for i in range(len(cfg_fc)-1):
            FCs.append(
                AttLIF(
                    p=dropout_ps[i+3],
                    attention='no',
                    inputSize=cfg_fc[i],
                    hiddenSize=cfg_fc[i+1],
                    spikeActFun=ActFun.apply,
                    useBatchNorm=False,
                    T=args.T,
                    pa_dict={'alpha': args.alpha, 'beta': args.beta, 'Vreset': args.Vreset, 'Vthres': args.Vthres},
                    reduction=args.reduction,
                    track_running_stats=args.track_running_stats,
                    mode_select=args.mode_select,
                    mem_act=args.mem_act,
                    TR_model=args.TR_model,
                )
            )
        self.FCs = nn.ModuleList(FCs)

    def forward(self, input, mp_collect=False):
        firing_nums = []
        if mp_collect:
            mps = []
        b, t, _, _, _ = input.size()
        outputs = input.permute(1, 0, 2, 3, 4).contiguous()

        for body_layer in self.ConvAttLIFs:
            outputs = body_layer(outputs, mp_collect=mp_collect)
            if mp_collect:
                firing_nums.append(outputs[0])
                mps.append(outputs[1])
                outputs = outputs[0]
            else:
                firing_nums.append(outputs)

        outputs = outputs.reshape(t, b, -1)

        for head_layer in self.FCs:
            outputs = head_layer(outputs, mp_collect=mp_collect)
            if mp_collect:
                firing_nums.append(outputs[0])
                mps.append(outputs[1])
                outputs = outputs[0]
            else:
                firing_nums.append(outputs)

        outputs = torch.mean(outputs.permute(1, 0, 2).contiguous(), dim=1)

        if mp_collect:
            return outputs, firing_nums, mps
        else:
            return outputs, firing_nums

