import torch
import torch.nn as nn
from torch.optim import lr_scheduler

import argparse
import copy

from vtrain import get_dataloader, get_optimizer, train, eval, net_save
from vmodel import vgg16_bn

#### HYPERPARAMETERS ####
epochs = 100

trained_root = "trained/"
if torch.cuda.is_available():
	device='cuda'
	print ("CUDA available; using GPU")
else:
	device='cpu'
dataset = "CIFAR-100"
download = True
batch_size = 32

trainloader, testloader = get_dataloader(dataset, download, batch_size)

criterion = nn.CrossEntropyLoss()

# for optimizer_type in ["SGD", "Adam", "RMSprop", "AdamW", "KFAC", "AggMo", "AdaBelief"]:
#     for ablate in [False, True]:
#         for scheduler_setting in ["sched", 0.1, 1e-2, 1e-3]:
#             for wd in [1e-4, 0]:

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--optim', type=str, required=True, help='optimizer type')
    parser.add_argument('--ablate', action='store_true', default=False, help='ablate BN or not')
    parser.add_argument('--lr', type=str, default="sched", help='learning rate')
    parser.add_argument('--wd', type=float, default=0, help='weight decay')
    args = parser.parse_args()

    optimizer_type = args.optim
    ablate = args.ablate
    scheduler_setting = args.lr
    wd = args.wd

    if scheduler_setting != "sched":
        scheduler_setting = float(scheduler_setting)

    suffix = "ablate{}_optim{}_lr{}_wd{}".format(str(ablate), optimizer_type, scheduler_setting, wd)

    # Train
    print("\n------------------ Training ------------------\n")
    print("Training: Ablate {}, Optimizer {}, LR Scheduler {}, Weight Decay {}".format(ablate, optimizer_type, scheduler_setting, wd))

    net = vgg16_bn().to(device)
    net_base = copy.deepcopy(net).to(device)

    if scheduler_setting == "sched":
        base_lr = 0.1
    else:
        base_lr = scheduler_setting

    optimizer = get_optimizer(net, opt_type=optimizer_type, lr=base_lr, wd=wd)

    if scheduler_setting == "sched":
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    accs_dict = {'Train': [], 'Test': [], 'lr': []}

    for epoch in range(epochs):
        train_acc = train(net, net_base, trainloader, optimizer, criterion, device, batch_size, epoch, ablate=ablate)

        if scheduler_setting == "sched":
            scheduler.step()

        test_acc = eval(net, testloader, criterion, device, epoch=epoch)

        accs_dict['Train'].append(train_acc)
        accs_dict['Test'].append(test_acc)
        accs_dict['lr'].append(optimizer.param_groups[0]['lr'])

        if (epoch+1) % 25 == 0:
            net_save(net, accs_dict, trained_root, suffix)
