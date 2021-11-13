import torch
import torch.nn as nn
import copy

from vtrain import get_dataloader, get_optimizer, FindLR, train, eval, net_save
from vmodel import vgg16_bn

#### HYPERPARAMETERS ####
epochs = 200

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
wd = 0.3
max_lr = 10
num_iter = 100

trainloader, testloader = get_dataloader(dataset, download, batch_size)

criterion = nn.CrossEntropyLoss()

if __name__ == "__main__":
    for optimizer_type in ["SGD", "Adam"]:
        for ablate in [True, False]:
            suffix = "ablate{}_optim{}".format(str(ablate), optimizer_type)

            # Train
            print("\n------------------ Training ------------------\n")
            print("Training: Ablate {}, Optimizer {}".format(ablate, optimizer_type))

            net = vgg16_bn().to(device)
            net_base = copy.deepcopy(net).to(device)

            optimizer = get_optimizer(net, opt_type=optimizer_type, lr=base_lr, wd=wd)
            lr_scheduler = FindLR(optimizer, max_lr=max_lr, num_iter=num_iter)

            accs_dict = {'Train': [], 'Test': []}

            for epoch in range(epochs):
                train_acc = train(net, net_base, trainloader, optimizer, criterion, device, batch_size, epoch, ablate=ablate)
                test_acc = eval(net, testloader, criterion, device, epoch=epoch)

                accs_dict['Train'].append(train_acc)
                accs_dict['Test'].append(test_acc)

                if epoch+1 % 25 == 0:
                    net_save(net, accs_dict, lr_scheduler.get_lr(), trained_root, suffix)
