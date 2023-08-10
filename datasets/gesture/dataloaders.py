import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from datasets.transforms import find_first, Repeat, toOneHot, ToTensor
from datasets.events_timeslices import chunk_evs_pol_dvs_gesture, get_tmad_slice


class DVSGestureDataset(Dataset):
    def __init__(
        self,
        root,
        train=True,
        transform=None,
        target_transform=None,
        clip=10,
        is_train_Enhanced=False,
        dt=1000,
        size=[2, 32, 32],
        ds=4,
        is_spike=False,
        interval_scaling=False,
        T=16,
    ):
        super(DVSGestureDataset, self).__init__()

        # self.n = 0
        self.root = root
        self.train = train
        self.chunk_size = T
        self.clip = clip
        self.is_train_Enhanced = is_train_Enhanced
        self.dt = dt
        self.transform = transform
        self.target_transform = target_transform
        self.size = size
        self.ds = ds

        self.is_spike = is_spike
        self.interval_scaling = interval_scaling

        if train:
            root_train = os.path.join(self.root, 'train')
            for _, _, self.files_train in os.walk(root_train):
                pass
            self.n = len(self.files_train)
        else:
            root_test = os.path.join(self.root, 'test')
            for _, _, self.files_test in os.walk(root_test):
                pass
            self.n = len(self.files_test)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        # Important to open and close in getitem to enable num_workers > 0
        if self.train:
            # assert idx < 1175
            root_test = os.path.join(self.root, 'train')

            with h5py.File(root_test + os.sep + self.files_train[idx], 'r', swmr=True, libver="latest") as f:
                data, target = sample_train(
                    f, 
                    T=self.chunk_size, 
                    is_train_Enhanced=self.is_train_Enhanced, 
                    dt=self.dt
                )

            # data = event_drop(data, resolution=[128 // self.ds[0], 128 // self.ds[1]])
            data = chunk_evs_pol_dvs_gesture(
                data=data,
                dt=self.dt,
                T=self.chunk_size,
                size=self.size,
                ds=self.ds
            )

            if self.is_spike:
                data = np.int64(data > 0)
            if self.interval_scaling:
                data = data / data.max()

            if self.transform is not None:
                data = self.transform(data)

            if self.target_transform is not None:
                target = self.target_transform(target)

            return data, target
        else:
            # assert idx < 288
            root_test = os.path.join(self.root, 'test')

            with h5py.File(root_test + os.sep + self.files_test[idx], 'r', swmr=True, libver="latest") as f:
                data, target = sample_test(
                    f,
                    T=self.chunk_size,
                    clip=self.clip,
                    dt=self.dt
                )

            data_temp = []
            target_temp = []
            for i in range(self.clip):

                if self.transform is not None:
                    temp = chunk_evs_pol_dvs_gesture(
                        data=data[i],
                        dt=self.dt,
                        T=self.chunk_size,
                        size=self.size,
                        ds=self.ds
                    )

                    if self.is_spike:
                        temp = np.int64(temp > 0)
                    if self.interval_scaling:
                        temp = temp / temp.max()

                    data_temp.append(self.transform(temp))

                if self.target_transform is not None:
                    target_temp.append(self.target_transform(target))

            data = torch.stack(data_temp)
            target = torch.stack(target_temp)

            return data, target


def sample_train(
    hdf5_file,
    T=60,
    dt=1000,
    is_train_Enhanced=False
):
    label = hdf5_file['labels'][()]

    tbegin = hdf5_file['times'][0]
    tend = np.maximum(0, hdf5_file['times'][-1] - T * dt)

    start_time = np.random.randint(tbegin, tend) if is_train_Enhanced else 0

    tmad = get_tmad_slice(
        hdf5_file['times'][()],
        hdf5_file['addrs'][()],
        start_time,
        T * dt
    )
    tmad[:, 0] -= tmad[0, 0]
    return tmad[:, [0, 3, 1, 2]], label


def sample_test(
    hdf5_file,
    T=60,
    clip=10,
    dt=1000
):
    label = hdf5_file['labels'][()]

    tbegin = hdf5_file['times'][0]
    tend = np.maximum(0, hdf5_file['times'][-1])

    tmad = get_tmad_slice(
        hdf5_file['times'][()],
        hdf5_file['addrs'][()],
        tbegin,
        tend - tbegin
    )
    tmad[:, 0] -= tmad[0, 0]

    start_time = tmad[0, 0]
    end_time = tmad[-1, 0]

    start_point = []
    if clip * T * dt - (end_time - start_time) > 0:
        overlap = int(
            np.floor((clip * T * dt - (end_time - start_time)) / clip))
        for j in range(clip):
            start_point.append(j * (T * dt - overlap))
            if start_point[-1] + T * dt > end_time:
                diff = start_point[-1] + T * dt - end_time
                start_point[-1] = start_point[-1] - diff
    else:
        overlap = int(
            np.floor(((end_time - start_time) - clip * T * dt) / clip))
        for j in range(clip):
            start_point.append(j * (T * dt + overlap))
            if start_point[-1] + T * dt > end_time:
                diff = start_point[-1] + T * dt - end_time
                start_point[-1] = start_point[-1] - diff

    temp = []
    for start in start_point:
        idx_beg = find_first(tmad[:, 0], start)
        idx_end = find_first(tmad[:, 0][idx_beg:], start + T * dt) + idx_beg
        temp.append(tmad[idx_beg:idx_end][:, [0, 3, 1, 2]])

    return temp, label


def create_datasets(
    root=None,
    train=True,
    ds=4,
    dt=1000,
    transform=None,
    target_transform=None,
    n_events_attention=None,
    clip=10,
    is_train_Enhanced=False,
    is_spike = False,
    interval_scaling = False,
    T=16,
):
    if isinstance(ds, int):
        ds = [ds, ds]

    size = [2, 128 // ds[0], 128 // ds[1]]

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

    if transform is None:
        transform = default_transform()

    if target_transform is None:
        target_transform = Compose([
            Repeat(T), toOneHot(11)
        ])

    dataset = DVSGestureDataset(
        root,
        train=train,
        transform=transform,
        target_transform=target_transform,
        is_train_Enhanced=is_train_Enhanced,
        dt=dt,
        size=size,
        ds=ds,
        is_spike=is_spike,
        interval_scaling=interval_scaling,
        T=T,
        clip=clip,
    )
    return dataset
