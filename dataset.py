import torch


def create_dataset(args):
    if args.dataset == "gesture":
        from datasets.gesture.dataloaders import create_datasets
    elif args.dataset == "gesture_sj":
        from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
        def create_datasets(root, train, **kwargs):
            return DVS128Gesture(root, train=train, data_type="frame", frames_number=args.T, split_by="number")
    elif args.dataset == "gait_night" or args.dataset == "gait_day":
        from datasets.gait.dataloaders import create_datasets
    elif args.dataset == "recogition":
        from datasets.recogition.dataloaders import create_datasets
    elif args.dataset == "action":
        from datasets.action.dataloaders import create_datasets

    args.train_dataset = create_datasets(
        args.data_path,
        train=True,
        is_train_Enhanced=args.is_train_Enhanced,
        ds=args.ds,
        dt=args.dt * 1000,
        is_spike=args.is_spike,
        interval_scaling=args.interval_scaling,
        T=args.T,
    )
    args.test_dataset = create_datasets(
        args.data_path,
        train=False,
        ds=args.ds,
        dt=args.dt * 1000,
        clip=args.clip,
        is_spike=args.is_spike,
        interval_scaling=args.interval_scaling,
        T=args.T,
    )

    # Data loader
    args.train_loader = torch.utils.data.DataLoader(
        args.train_dataset,
        batch_size=args.batch_size_train,
        shuffle=True,
        drop_last=args.drop_last,
        num_workers=args.num_work,
        pin_memory=args.pip_memory,
    )
    args.test_loader = torch.utils.data.DataLoader(
        args.test_dataset,
        batch_size=args.batch_size_test,
        shuffle=False,
        drop_last=args.drop_last,
        num_workers=args.num_work,
        pin_memory=args.pip_memory
    )

