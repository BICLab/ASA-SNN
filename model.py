import torch


def create_net(args):
    if args.arch == "A":
        from models.Net import DVSGestureNet
    elif args.arch == "B":
        from models.Net2 import DVSGestureNet
    elif args.arch == "C":
        from models.Net3 import DVSGestureNet
    elif args.arch == "D":
        from models.Net4 import DVSGestureNet

    model = DVSGestureNet(args).to(args.device)

    args.model = model
    # args.model = nn.DataParallel(
    #     args.model,
    #     device_ids=args.device_ids
    # )

    args.optimizer = torch.optim.Adam(
        args.model.parameters(),
        lr=args.lr,
        betas=args.betas,
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    # args.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer=args.optimizer,
    #     mode='min',
    #     factor=0.2,
    #     patience=5,
    #     verbose=True
    # )
    args.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer=args.optimizer,
        T_0=args.num_epochs,
        T_mult=1,
        eta_min=1e-6,
        verbose=True,
    )
