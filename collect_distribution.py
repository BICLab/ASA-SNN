import os, time
import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from spikingjelly.clock_driven import functional

from model import create_net
from dataset import create_dataset
from config import parser
from utils import set_seed


@torch.no_grad()
def firing_num(args):
    args.model.load_state_dict(torch.load(args.modelPath))
    args.model.eval()

    save_name_file = os.path.join(args.name, "S_Feature")
    args.path = save_name_file
    if not os.path.exists(save_name_file):
        os.makedirs(save_name_file)
    f = h5py.File(os.path.join(save_name_file, 'all.h5'), 'w')

    all_idx = 0
    flag = True

    for n_iter, (input, labels) in enumerate(args.test_loader):
        functional.reset_net(args.model)

        b = input.size()[0]

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
                labels.size()[2])
            labels = labels.float().to(args.device)
        else:
            labels = labels.reshape(
                b * args.clip,
                labels.size()[2],
                labels.size()[3]
            )
            labels = labels[:, 1, :].float().to(args.device)

        output, firing_nums, mps = args.model(input=input, mp_collect=True)

        for i in range(len(firing_nums)):
            if i < 3:
                firing_num_mat = firing_nums[i].reshape(
                    b * args.clip, -1,
                    firing_nums[i].size()[2],
                    firing_nums[i].size()[3],
                    firing_nums[i].size()[4],
                )
                # print(firing_num_mat.shape)
                f.create_dataset(str(all_idx) +  '_S_' + str(i), data=firing_num_mat.squeeze().cpu().numpy())

        if flag:
            if args.arch == "A":
                layer = 3
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
    
        all_idx += 1

    g = h5py.File(os.path.join(args.path, 'all.h5'), 'r')
    f = h5py.File(os.path.join(args.path, 'mean.h5'), 'w')

    names = []
    data_all = []
    for pre in ['S']:
        for nex in range(layer):
            data_all.append([])
            names.append(pre + '_' + str(nex))

    for name in names:
        start_time = time.time()
        X = []
        for idx in range(len(args.test_loader)):
            if idx == 0:
                 X = g[str(idx) + '_' + name][()]
            else:
                X = X + g[str(idx) + '_' + name][()]
            # X.append(g[str(idx) + '_' + name][()])
        # X = np.array(X)
        f.create_dataset(name, data= X / len(args.test_dataset))
        # f1.create_dataset(name, data=np.std(X, axis=0))
        print('costs:', time.time() - start_time)

    f.close()


@torch.no_grad()
def mp_num(args):
    args.model.load_state_dict(torch.load(args.modelPath))
    args.model.eval()

    save_name_file = os.path.join(args.name, "S_mp")
    args.path = save_name_file
    if not os.path.exists(save_name_file):
        os.makedirs(save_name_file)
    f = h5py.File(os.path.join(save_name_file, 'all.h5'), 'w')

    all_idx = 0
    flag = True

    for n_iter, (input, labels) in enumerate(args.test_loader):
        functional.reset_net(args.model)

        b = input.size()[0]

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
                labels.size()[2])
            labels = labels.float().to(args.device)
        else:
            labels = labels.reshape(
                b * args.clip,
                labels.size()[2],
                labels.size()[3]
            )
            labels = labels[:, 1, :].float().to(args.device)

        output, _, mps = args.model(input=input, mp_collect=True)

        if flag:
            if args.arch == "A":
                layer = 3
            elif args.arch == "B":
                layer = 5
            flag = False

        for i in range(len(mps)):
            if i < layer:
                mp = mps[i].reshape(
                    b * args.clip, -1,
                    mps[i].size()[2],
                    mps[i].size()[3],
                    mps[i].size()[4],
                )
                # print(np.sum(mp.cpu().numpy()>0.3) / mp.numel())
                f.create_dataset(str(all_idx) +  '_S_' + str(i), data=mp.squeeze().cpu().numpy())

        _, predicted = torch.max(output.data, 1)
        _, labelTest = torch.max(labels.data, 1)
        for i in range(b):
            predicted_clips = predicted[i * args.clip:(i + 1) * args.clip]
            labelTest_clips = labelTest[i * args.clip:(i + 1) * args.clip]
            test_clip_correct = (predicted_clips == labelTest_clips).sum().item()
            if test_clip_correct / args.clip > 0.5:
                args.test_correct += 1
        args.test_acc = 100. * float(args.test_correct) / float(n_iter+1)
        print("iteration: {}/{}\tacc: {}".format(n_iter + 1, len(args.test_dataset), args.test_acc))

        all_idx += 1

    g = h5py.File(os.path.join(args.path, 'all.h5'), 'r')
    f = h5py.File(os.path.join(args.path, 'mean.h5'), 'w')

    names = []
    data_all = []
    for pre in ['S']:
        for nex in range(layer):
            data_all.append([])
            names.append(pre + '_' + str(nex))

    for name in names:
        start_time = time.time()
        X = []
        for idx in range(len(args.test_dataset)):
            if idx == 0:
                 X = g[str(idx) + '_' + name][()]
            else:
                X = X + g[str(idx) + '_' + name][()]
            # X.append(g[str(idx) + '_' + name][()])
        # X = np.array(X)
        # print(np.sum(X / len(args.test_dataset)>=0.3) / np.size(X))
        f.create_dataset(name, data=X / len(args.test_dataset))
        # f1.create_dataset(name, data=np.std(X, axis=0))
        print('costs:', time.time() - start_time)

    f.close()


def mean_std_firing(args):
    # args.path = os.path.join(args.name, "S_Feature")
    # print(os.path.join(args.path, args.mean_std + '.h5'))
    f = h5py.File(os.path.join(args.path, args.mean_std + '.h5'), 'r')
    save_path_mean = args.path + os.sep + args.mean_std

    save_path_mean_name = os.path.join(save_path_mean, args.name)
    if not os.path.exists(save_path_mean_name):
        os.makedirs(save_path_mean_name)
    data_ = f[args.feature_name][()]
    f.close()

    _, T, C, _, _ = data_.shape

    for t in range(T):
        for c in range(C):
            data = np.flipud(data_[0, t, c, :, :][()].T)
            # print(data.shape)
            if data.shape[0] == 32:
                fig = plt.figure(figsize=(50, 50), clear=True)
            elif data.shape[0] == 16:
                fig = plt.figure(figsize=(25, 25), clear=True)
            elif data.shape[0] == 8:
                fig = plt.figure(figsize=(12, 12), clear=True)
            else:
                fig = plt.figure(figsize=(5, 5), clear=True)

            ax = sns.heatmap(
                data=data,
                annot=True,
                fmt='.2f',
                cbar=False,
                cmap="RdBu_r",
                vmin = data.min(),
                vmax = data.max(),
            )
            plt.xlabel('max:' + str(data.max()) + '_min:' + str(data.min()))
            plt.tight_layout()
            plt.title('C_' + str(c) + 'T_' + str(t))
            # plt.show()
            plt.savefig(
                os.path.join(save_path_mean_name, 'C_' + str(c) + 'T_' + str(t) + '.png'),
                bbox_inches='tight', pad_inches=0
            )

            plt.clf()
            plt.cla()
            plt.close(fig)


def mean_std_mp(args):
    # args.path = os.path.join(args.name, "S_Feature")
    # print(os.path.join(args.path, args.mean_std + '.h5'))
    f = h5py.File(os.path.join(args.path, args.mean_std + '.h5'), 'r')
    save_path_mean = args.path + os.sep + args.mean_std

    save_path_mean_name = os.path.join(save_path_mean, args.feature_name)
    if not os.path.exists(save_path_mean_name):
        os.makedirs(save_path_mean_name)
    data_ = f[args.feature_name][()]
    f.close()

    _, T, C, _, _ = data_.shape

    spike = []
    for t in range(T):
        for c in range(C):
            data = np.flipud(data_[0, t, c, :, :][()].T)
            spike.append(np.sum(data>=0.3) / np.size(data))

            fig = plt.figure(figsize=(5, 5), clear=True)

            sns.distplot(data, bins=50, kde=False)
            plt.xlabel('max:' + str(data.max()) + '_min:' + str(data.min()))
            plt.tight_layout()
            plt.title('C_' + str(c) + 'T_' + str(t))
            # plt.show()
            plt.savefig(
                os.path.join(save_path_mean_name, 'C_' + str(c) + 'T_' + str(t) + '.png'),
                bbox_inches='tight', pad_inches=0
            )
            print('C_' + str(c) + 'T_' + str(t))

            plt.clf()
            plt.cla()
            plt.close(fig)


def main():
    args = parser.parse_args()

    set_seed(args.seed)
    
    args.test_correct = 0.
    args.recordPath = args.modelPath
    args.im_width, args.im_height = (128 // args.ds, 128 // args.ds)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device_ids = range(torch.cuda.device_count())

    # args.name = args.dataset + '_dt=' + str(args.dt) + 'ms' + '_T=' + str(args.T) + '_attn=' + args.attention
    args.name = args.dataset + '_dt=' + str(args.dt) + 'ms' + '_T=' + str(args.T) + '_attn=' + args.attention + '_seed=' + str(args.seed) + '_arch=' + args.arch
    args.modelNames = args.name + '.pth'
    args.recordNames = args.name + '.csv'

    print(args)

    create_dataset(args=args)

    create_net(args=args)

    if args.collect == "firing":
        firing_num(args=args)
        # args.mean_std = "mean"
        # mean_std_firing(args=args)
    elif args.collect == "mp":
        mp_num(args=args)
        # args.mean_std = "mean"
        # mean_std_mp(args=args)

if __name__ == '__main__':
    main()

