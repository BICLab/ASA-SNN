import math
import torch
from torchinfo import summary

from model import create_net
from dataset import create_dataset
from config import parser
from apis import run
from utils import set_seed, save_csv


def main():
    args = parser.parse_args()

    set_seed(args.seed)

    args.recordPath = args.modelPath
    args.im_width, args.im_height = (128 // args.ds, 128 // args.ds)
    if args.dataset == "action" or args.dataset == "recogition":
        args.im_width, args.im_height = (math.ceil(346 / args.ds), 260 // args.ds)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device_ids = range(torch.cuda.device_count())
    args.modelPath = args.modelPath + args.attention
    args.name = args.dataset + '_dt=' + str(args.dt) + 'ms' + '_T=' + str(args.T) + '_attn=' + args.attention + '_reduc=' + str(args.reduction)  + '_lam=' + str(args.lam) + '_seed=' + str(args.seed) + '_arch=' + str(args.arch)
    args.modelNames = args.name + '.pth'
    args.recordNames = args.name + '.csv'

    print(args.name)

    create_dataset(args=args)

    create_net(args=args)

    summary(args.model, (2, args.T, args.in_channels, args.im_height, args.im_width), depth=3)

    run(args=args)

    print('best acc:', args.best_acc, 'best_epoch:', args.best_epoch)

    save_csv(args=args)


if __name__ == '__main__':
    main()
