import os
import torch

from apis.train import train
from apis.test import test


def run(args):
    args.best_acc = 0
    args.best_epoch = 0

    args.epoch_list = []
    args.loss_train_list = []
    args.loss_test_list = []
    args.acc_train_list = []
    args.acc_test_list = []

    if args.pretrained_path != None:
        pre_dict = torch.load(args.pretrained_path)['net']
        pre = {}
        for k, _ in pre_dict.items():
            pre[k[7:]] = pre_dict[k]
        model_dict = args.model.state_dict()
        pre_dict = {k: v for k, v in pre.items() if k in model_dict}
        # print(pre_dict.keys())
        model_dict.update(pre_dict)
        args.model.load_state_dict(model_dict)
        print('loading model...')

    for args.epoch in range(args.num_epochs):
        args.model.train()
        train(args=args)

        args.train_loss = args.train_loss / len(args.train_loader)
        args.epoch_list.append(args.epoch + 1)
        args.train_acc = 100. * float(args.train_correct) / float(len(args.train_dataset))
        print('epoch:', args.epoch + 1)
        print('dt:', args.dt)
        print('T:', args.T)
        print('Tarin loss:%.5f' % args.train_loss)
        print('Train acc: %.3f' % args.train_acc)

        if args.lr_scheduler:
            args.scheduler.step(args.epoch)

        args.loss_train_list.append(args.train_loss)
        args.acc_train_list.append(args.train_acc)

        # test
        with torch.no_grad():
            args.model.eval()
            test(args=args)

            args.test_loss = args.test_loss / len(args.test_loader)
            args.test_acc = 100. * float(args.test_correct) / float(len(args.test_dataset))
            args.loss_test_list.append(args.test_loss)
            print('Test loss:%.5f' % args.test_loss)
            print('Test acc: %.3f' % args.test_acc)

            args.acc_test_list.append(args.test_acc)

            if args.test_acc >= args.best_acc:
                args.best_epoch = args.epoch + 1
                args.best_acc = args.test_acc

                print('Saving..')

                if not os.path.exists(args.modelPath):
                    os.makedirs(args.modelPath)
                torch.save(
                    args.model.state_dict(),
                    args.modelPath + os.sep + args.modelNames,
                )

            print('best acc:', args.best_acc)

