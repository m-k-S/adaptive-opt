import torch
import torch.nn as nn
from torch.optim import lr_scheduler

import argparse
import copy

from vtrain import get_dataloader, get_optimizer, train, eval, net_save
from vmodel import vgg16_bn

#### HYPERPARAMETERS ####
epochs = 75

trained_root = "trained/"
if torch.cuda.is_available():
	device='cuda'
	print ("CUDA available; using GPU")
else:
	device='cpu'
dataset = "CIFAR-10"
download = True
batch_size = 32

trainloader, testloader = get_dataloader(dataset, download, batch_size)

criterion = nn.CrossEntropyLoss()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--optim', type=str, required=True, help='optimizer type')
    parser.add_argument('--ablate', action='store_true', default=False, help='ablate BN or not')
    parser.add_argument('--lr', type=str, default="sched", help='learning rate')
    parser.add_argument('--wd', type=float, default=0, help='weight decay')
    parser.add_argument('--mom', type=float, default=0, help='momentum')
    args = parser.parse_args()

    optimizer_type = args.optim
    ablate = args.ablate
    scheduler_setting = args.lr
    wd = args.wd
    mom = args.mom

    if scheduler_setting != "sched":
        scheduler_setting = float(scheduler_setting)

    suffix = "ablate{}_optim{}_lr{}_wd{}_mom{}".format(str(ablate), optimizer_type, scheduler_setting, wd, mom)

    # Train
    print("\n------------------ Training ------------------\n")
    print("Training: Ablate {}, Optimizer {}, LR Scheduler {}, Weight Decay {}, Momentum {}".format(ablate, optimizer_type, scheduler_setting, wd, mom))

    net = vgg16_bn().to(device)
    net_base = copy.deepcopy(net).to(device)

    if scheduler_setting == "sched":
        base_lr = 0.1
    else:
        base_lr = scheduler_setting

    optimizer = get_optimizer(net, opt_type=optimizer_type, lr=base_lr, wd=wd, mom=mom)

    if scheduler_setting == "sched":
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    accs_dict = {'Train': [], 'Test': [], 'lr': [], 'train_loss': [], 'test_loss': [], 'param_norms': []}

    for epoch in range(epochs):
        train_acc, train_loss = train(net, net_base, trainloader, optimizer, criterion, device, batch_size, epoch, ablate=ablate)

        if scheduler_setting == "sched":
            scheduler.step()

        test_acc, test_loss = eval(net, testloader, criterion, device, epoch=epoch)

        accs_dict['Train'].append(train_acc)
        accs_dict['Test'].append(test_acc)
        accs_dict['lr'].append(optimizer.param_groups[0]['lr'])
        accs_dict['train_loss'].append(train_acc)
        accs_dict['test_loss'].append(test_loss)

        if (epoch+1) % 25 == 0:
            net_save(net, accs_dict, trained_root, suffix, ablate)
