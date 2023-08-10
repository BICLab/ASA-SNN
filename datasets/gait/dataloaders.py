import os
import random

import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from datasets.transforms import find_first, Repeat, toOneHot, ToTensor
from datasets.events_timeslices import chunk_evs_pol_dvs_gait, get_tmad_slice
from datasets.event_drop import event_drop


class DVS128GaitDataset(Dataset):
    def __init__(
        self,
        path,
        dt=1000,
        T=10,
        train=True,
        is_train_Enhanced=False,
        clips=1,
        is_spike=False,
        ds=None,
    ):
        super(DVS128GaitDataset, self).__init__()
        if ds is None:
            ds = [1, 1]
        self.train = train
        self.dt = dt
        self.T = T
        self.is_train_Enhanced = is_train_Enhanced
        self.clips = clips
        self.is_spike = is_spike
        self.ds = ds

        if self.train:
            train_npy_path = os.path.join(path, 'train')
            self.train_data = np.load(os.path.join(train_npy_path, 'train_data.npy'), allow_pickle=True)
            self.train_target = np.load(os.path.join(train_npy_path, 'train_target.npy'), allow_pickle=True)
        else:
            test_npy_path = os.path.join(path, 'test')
            self.test_data = np.load(os.path.join(test_npy_path, 'test_data.npy'), allow_pickle=True)
            self.test_target = np.load(os.path.join(test_npy_path, 'test_target.npy'), allow_pickle=True)

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def __getitem__(self, idx):
        if self.train:
            data = self.train_data[idx]
            time_all = data[-1][0] - data[0][0]

            data = sample_train(
                data=data,
                dt=self.dt,
                T=self.T,
                is_train_Enhanced=self.is_train_Enhanced,
            )
            data = event_drop(data, resolution=[128 // self.ds[0], 128 // self.ds[1]])

            data = chunk_evs_pol_dvs_gait(
                data=data,
                dt=self.dt,
                T=self.T,
                ds=self.ds
            )

            if self.is_spike:
                data = np.int64(data > 0)

            target_idx = self.train_target[idx]
            label = np.zeros((20))
            label[target_idx] = 1.0

            return data, label
        else:
            data = self.test_data[idx]
            time_all = data[-1][0] - data[0][0]

            data = sample_test(
                data=data,
                dt=self.dt,
                T=self.T,
                clips=self.clips,
            )

            target_idx = self.test_target[idx]
            label = np.zeros((20))
            label[target_idx] = 1.0

            data_temp = []
            target_temp = []
            for i in range(self.clips):
                temp = chunk_evs_pol_dvs_gait(
                    data=data[i],
                    dt=self.dt,
                    T=self.T,
                    ds=self.ds
                )

                if self.is_spike:
                    temp = np.int64(temp > 0)

                data_temp.append(temp)

                target_temp.append(label)

            data = np.array(data_temp)
            target = np.array(target_temp)

            return data, target


def sample_train(
    data,
    T=60,
    dt=1000,
    is_train_Enhanced=False
):
    tbegin = data[:, 0][0]
    tend = np.maximum(0, data[:, 0][-1] - T * dt)

    start_time = random.randint(tbegin, tend) if is_train_Enhanced else tbegin

    tmad = get_tmad_slice(
        data[:, 0],
        data[:, 1:4],
        start_time,
        T * dt
    )
    if len(tmad) == 0:
        return tmad
    tmad[:, 0] -= tmad[0, 0]
    return tmad


def sample_test(
    data,
    T=60,
    clips=10,
    dt=1000
):
    tbegin = data[:, 0][0]
    tend = np.maximum(0, data[:, 0][-1])

    tmad = get_tmad_slice(
        data[:, 0],
        data[:, 1:4],
        tbegin,
        tend - tbegin
    )
    # 初试从零开始
    tmad[:, 0] -= tmad[0, 0]

    start_time = tmad[0, 0]
    end_time = tmad[-1, 0]

    start_point = []
    if clips * T * dt - (end_time - start_time) > 0:
        overlap = int(
            np.floor((clips * T * dt - (end_time - start_time)) / clips))
        for j in range(clips):
            start_point.append(j * (T * dt - overlap))
            if start_point[-1] + T * dt > end_time:
                diff = start_point[-1] + T * dt - end_time
                start_point[-1] = start_point[-1] - diff
    else:
        overlap = int(
            np.floor(((end_time - start_time) - clips * T * dt) / clips))
        for j in range(clips):
            start_point.append(j * (T * dt + overlap))
            if start_point[-1] + T * dt > end_time:
                diff = start_point[-1] + T * dt - end_time
                start_point[-1] = start_point[-1] - diff

    temp = []
    for start in start_point:
        idx_beg = find_first(tmad[:, 0], start)
        idx_end = find_first(tmad[:, 0][idx_beg:], start + T * dt) + idx_beg
        temp.append(tmad[idx_beg:idx_end])

    return temp


def create_datasets(
    root=None,
    train=True,
    chunk_size_train=60,
    chunk_size_test=60,
    ds=4,
    dt=1000,
    transform_train=None,
    transform_test=None,
    target_transform_train=None,
    target_transform_test=None,
    n_events_attention=None,
    clip=10,
    is_train_Enhanced=False,
    is_spike=False,
    interval_scaling=False,
    T=16,
):
    if isinstance(ds, int):
        ds = [ds, ds]

    if n_events_attention is None:
        def default_transform():
            return Compose([
                ToTensor()
            ])
    else:
        def default_transform():
            return Compose([
                ToTensor()
            ])

    if transform_train is None:
        transform_train = default_transform()
    if transform_test is None:
        transform_test = default_transform()

    if target_transform_train is None:
        target_transform_train = Compose(
            [Repeat(chunk_size_train), toOneHot(11)])
    if target_transform_test is None:
        target_transform_test = Compose(
            [Repeat(chunk_size_test), toOneHot(11)])

    dataset = DVS128GaitDataset(
        root,
        dt=dt,
        T=T,
        train=train,
        is_train_Enhanced=is_train_Enhanced,
        clips=clip,
        is_spike=is_spike,
        ds=ds,
    )
    return dataset
