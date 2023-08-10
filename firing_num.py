import os
import torch
import torch.nn.functional as F

import numpy as np
import pandas as pd
from spikingjelly.clock_driven import functional

from model import create_net
from dataset import create_dataset
from config import parser
from utils import set_seed


@torch.no_grad()
def firing_num(args):
    args.model.load_state_dict(torch.load(args.modelPath))
    args.model.eval()

    all_idx = 0
    flag = True

    for n_iter, (input, labels) in enumerate(args.test_loader):
        functional.reset_net(args.model)

        b = input.size()[0]

        if "sj" not in args.dataset:
            input = input.reshape(
                b * args.clip,
                input.size()[2],
                input.size()[3],
                input.size()[4],
                input.size()[5]
            )
            input = input.float().to(args.device)
            if len(labels.shape) == 3:
                labels = labels.reshape(
                    b * args.clip,
                    labels.size()[2]
                )
                labels = labels.float().to(args.device)
            else:
                labels = labels.reshape(
                    b * args.clip,
                    labels.size()[2],
                    labels.size()[3]
                )
                labels = labels[:, 1, :].float().to(args.device)
        else:
            input = input.float().to(args.device)
            if len(labels.shape) == 2:
                labels = labels.float().to(args.device)
            elif len(labels.shape) == 1:
                labels = F.one_hot(labels, args.num_classes).float().to(args.device)

        output, firing_num = args.model(input)

        if flag:
            neural_list = []
            for f in firing_num:
                neural_list.append(f.numel() / f.shape[0] / f.shape[1])
            flag = False

        _, predicted = torch.max(output.data, 1)
        _, labelTest = torch.max(labels.data, 1)
        for i in range(b):
            predicted_clips = predicted[i * args.clip:(i + 1) * args.clip]
            labelTest_clips = labelTest[i * args.clip:(i + 1) * args.clip]
            test_clip_correct = (predicted_clips == labelTest_clips).sum().item()
            if test_clip_correct / args.clip > 0.5:
                args.test_correct += 1
        args.test_acc = 100. * float(args.test_correct) / (float(n_iter+1) * b)
        print("iteration: {}/{}\tacc: {}".format(n_iter + 1, len(args.test_loader), args.test_acc))

        list_ = []
        for firing_single in firing_num:
            sub_list = []
            firing_single = firing_single.cpu().detach().numpy()
            for T_ in range(args.T):
                sub_list.append(np.sum(firing_single[T_, ...]))
            sub_list.append(torch.from_numpy(firing_single[0, ...]).numel())
            list_.append(sub_list)

        csv = pd.DataFrame(
            data=list_
        )
        if not os.path.exists(args.name):
            os.makedirs(args.name)
        csv.to_csv(args.name + os.sep + str(all_idx) + '.csv')
        all_idx += 1

    flag = True
    spiking_all = []
    for idx in range(len(args.test_loader)):
        name = str(idx) + '.csv'
        df = pd.read_csv(os.path.join(args.name, name), header=None).values

        if flag:
            for layer in range(len(df)-1):
                spiking_all.append(df[layer+1 , 1:])
            flag = False
        else:
            for layer in range(len(df)-1):
                spiking_all[layer] = spiking_all[layer] + df[layer+1 , 1:]
    
    firing_nums = []
    for nums in spiking_all:
        sub_list = []
        num = 0
        total = 0
        for idx in range(len(nums)-1):
            num += nums[idx]
            total += nums[len(nums)-1]
            sub_list.append(nums[idx] / nums[len(nums) - 1])
        sub_list.append(num / total)
        firing_nums.append(sub_list)

    csv = pd.DataFrame(
        data=firing_nums
    )
    csv.to_csv(args.name+".csv")

    total_spike = 0.
    for neural, firing_num in zip(neural_list, firing_nums):
        total_spike += firing_num[-1] * neural
    print(total_spike / sum(neural_list))


def main(i):
    args = parser.parse_args()

    set_seed(args.seed)
    
    args.test_correct = 0.
    args.recordPath = args.modelPath
    args.im_width, args.im_height = (128 // args.ds, 128 // args.ds)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device_ids = range(torch.cuda.device_count())

    args.name = args.dataset + '_dt=' + str(args.dt) + 'ms' + '_T=' + str(args.T) + '_attn=' + args.attention + '_lam=' + str(args.lam) + '_seed=' + str(args.seed) + '_arch=' + str(args.arch) + "_" + str(i)
    args.modelNames = args.name + '.pth'
    args.modelPath = os.path.join(os.path.join(args.modelPath, args.attention), args.modelNames)
    args.recordNames = args.name + '.csv'

    print(args)

    create_dataset(args=args)

    create_net(args=args)

    firing_num(args=args)

if __name__ == '__main__':
    main(0)
    main(1)

