import os
import math
import numpy as np
import torch
from torch.utils.data import Dataset

from datasets.transforms import find_first, Repeat, toOneHot, ToTensor
from datasets.events_timeslices import chunk_evs_pol_dvs_gesture, get_tmad_slice
from torchvision.transforms import Compose, RandomCrop, CenterCrop


mapping = {
    'arm crossing': 0,
    'turning around': 1,
    'walking': 2,
    'picking up': 3,
    'throwing': 4,
    'get-up': 5,
    'jumping': 6,
    'kicking': 7,
    'sit-down': 8,
    'waving': 9,
}


class DVSRecognitionDataset_Recogition(Dataset):
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
        super(DVSRecognitionDataset_Recogition, self).__init__()

        self.n = 0
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
            self.files_train=[]
            for self.now_file, self.label_train_one, self.files_train_one in os.walk(root_train):
                for now_data_file in self.files_train_one:
                    if now_data_file[-3:]=='npy':
                        self.files_train.append(os.path.join(self.now_file,now_data_file))
            self.n = len(self.files_train)
        else:
            root_test = os.path.join(self.root, 'test')
            self.files_test = []
            for self.now_file, self.label_train_one, self.files_train_one in os.walk(root_test):
                for now_data_file in self.files_train_one:
                    if now_data_file[-3:] == 'npy':
                        self.files_test.append(os.path.join(self.now_file,now_data_file))
            self.n = len(self.files_test)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):

        # Important to open and close in getitem to enable num_workers>0

        if self.train:
            root_test = os.path.join(self.root, 'train')

            file_path = self.files_train[idx]
            data, target, time_all = sample_train(
                file_path, 
                T=self.chunk_size, 
                is_train_Enhanced=self.is_train_Enhanced, 
                dt=self.dt,
            )

            data = chunk_evs_pol_dvs_gesture(
                data=data,
                dt=self.dt,
                T=self.chunk_size,
                size=self.size,
                ds=self.ds
            )
            if self.is_spike:
                data = np.int64(data>0)
            if self.interval_scaling:
                data = data / data.max()

            if self.transform is not None:
                data = self.transform(data)

            if self.target_transform is not None:
                target = self.target_transform(target)

            return data, target
        else:
            file_path = self.files_test[idx]
            data, target, time_all = sample_test(
                file_path,
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
                        temp = np.int64(temp>0)

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
    is_train_Enhanced=False,
):
    label = mapping[hdf5_file.split('/')[-2]]
    data_dvs = np.load(hdf5_file)
    time_all = data_dvs[0][-1] - data_dvs[0][0]
    
    tbegin = data_dvs[0][-1]
    if data_dvs[0][-1] - tbegin < T*dt:
        start_time = tbegin
    else:
        tend = np.maximum(0, data_dvs[0][-1] - T * dt)
        # print('tend:',tend,'tbegin:',tbegin,'hdf5_file:',hdf5_file)
        try:
            start_time = np.random.randint(tbegin, tend) if is_train_Enhanced else 0
        except:
            print()

    tmad = get_tmad_slice(
        data_dvs[0],
        data_dvs[1:,:].T,
        start_time,
        T * dt
    )
    try:
        tmad[:, 0] -= tmad[0, 0]
    except Exception as e:
        pass
    return tmad[:, [0, 3, 1, 2]], label, time_all


def sample_test(
    hdf5_file,
    T=60,
    clip=10,
    dt=1000
):
    label = mapping[hdf5_file.split('/')[-2]]
    data_dvs = np.load(hdf5_file)
    tbegin = data_dvs[0][0]
    time_all = data_dvs[0][-1] - data_dvs[0][0]
    tend = np.maximum(0, data_dvs[0][-1])

    tmad = get_tmad_slice(
        data_dvs[0],
        data_dvs[1:, :].T,
        tbegin,
        tend-tbegin
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

    return temp, label, time_all


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

    size = [2, math.ceil(346 / ds[0]), 260 // ds[1]]

    if n_events_attention is None:
        def default_transform():
            return Compose([
                ToTensor(),
                # Resize(
                #     size=(128, 128),  # (height, width)
                #     interpolation=2  # 插值方法，一般保持默认就好
                # )
                # CenterCrop(128)
            ])
    else:
        def default_transform():
            return Compose([
                ToTensor(),
                # Resize(
                #     size=(128, 128),  # (height, width)
                #     interpolation=2  # 插值方法，一般保持默认就好
                # )
                # CenterCrop(128)
            ])

    if transform is None:
        transform = default_transform()

    if target_transform is None:
        target_transform = Compose([
            Repeat(T), toOneHot(10)
        ])

    dataset = DVSRecognitionDataset_Recogition(
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

