import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import copy

from vtrain import get_dataloader, get_optimizer, FindLR, train, eval, net_save
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

base_lr = 0.1
wd = 1e-4

trainloader, testloader = get_dataloader(dataset, download, batch_size)

criterion = nn.CrossEntropyLoss()

if __name__ == "__main__":
    for optimizer_type in ["SGD", "Adam"]:
        for ablate in [False, True]:
            suffix = "ablate{}_optim{}".format(str(ablate), optimizer_type)

            # Train
            print("\n------------------ Training ------------------\n")
            print("Training: Ablate {}, Optimizer {}".format(ablate, optimizer_type))

            net = vgg16_bn().to(device)
            net_base = copy.deepcopy(net).to(device)

            optimizer = get_optimizer(net, opt_type=optimizer_type, lr=base_lr, wd=wd)
            scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

            accs_dict = {'Train': [], 'Test': [], 'lr': []}

            for epoch in range(epochs):
                train_acc = train(net, net_base, trainloader, optimizer, criterion, device, batch_size, epoch, ablate=ablate)
                scheduler.step()
                test_acc = eval(net, testloader, criterion, device, epoch=epoch)

                accs_dict['Train'].append(train_acc)
                accs_dict['Test'].append(test_acc)
                accs_dict['lr'].append(optimizer.param_groups[0]['lr'])

                if (epoch+1) % 25 == 0:
                    net_save(net, accs_dict, trained_root, suffix)
