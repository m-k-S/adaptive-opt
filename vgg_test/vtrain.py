import time
import pickle as pkl

import torch
import torch.nn as nn
import torchvision

from torchvision import models, transforms
from torch.optim import SGD, Adam, RMSprop, AdamW
from torch.optim.lr_scheduler import _LRScheduler

from aggmo import AggMo
from adabelief import AdaBelief
from kfac import KFACOptimizer


class FindLR(_LRScheduler):
    """exponentially increasing learning rate
    Args:
        optimizer: optimzier(e.g. SGD)
        num_iter: totoal_iters
        max_lr: maximum  learning rate
    """
    def __init__(self, optimizer, max_lr=10, num_iter=100, last_epoch=-1):
        self.total_iters = num_iter
        self.max_lr = max_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * (self.max_lr / base_lr) ** (self.last_epoch / (self.total_iters + 1e-32)) for base_lr in self.base_lrs]


def get_dataloader(use_data, download, bsize):
	######### Dataloaders #########
	transform = transforms.Compose(
		[transforms.RandomHorizontalFlip(),
		 transforms.ToTensor(),
		 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
		 ])
	transform_test = transforms.Compose(
		[transforms.ToTensor(),
		 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
		 ])

	d_path = "../"
	if(use_data=="CIFAR-10"):
		n_classes = 10
		trainset = torchvision.datasets.CIFAR10(root=d_path+'datasets/cifar10/', train=True, download=(download), transform=transform)
		trainloader = torch.utils.data.DataLoader(trainset, batch_size=bsize, shuffle=True, num_workers=2)
		testset = torchvision.datasets.CIFAR10(root=d_path+'datasets/cifar10/', train=False, download=(download), transform=transform_test)
		testloader = torch.utils.data.DataLoader(testset, batch_size=bsize, shuffle=False, num_workers=2)
	elif(use_data=="CIFAR-100"):
		n_classes = 100
		trainset = torchvision.datasets.CIFAR100(root=d_path+'datasets/cifar100/', train=True, download=(download), transform=transform)
		trainloader = torch.utils.data.DataLoader(trainset, batch_size=bsize, shuffle=True, num_workers=2)
		testset = torchvision.datasets.CIFAR100(root=d_path+'datasets/cifar100/', train=False, download=(download), transform=transform_test)
		testloader = torch.utils.data.DataLoader(testset, batch_size=bsize, shuffle=False, num_workers=2)
	else:
		raise Exception("Not CIFAR-10/CIFAR-100")

	return trainloader, testloader

def get_optimizer(net, lr, wd, opt_type="SGD"):
    if opt_type == "SGD":
        optimizer = SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    elif opt_type == "RMSProp":
        optimizer = RMSprop(net.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    elif opt_type == "Adam":
        optimizer = Adam(net.parameters(), lr=lr, weight_decay=wd)
    elif opt_type == "AdamW":
        optimizer = AdamW(net.parameters(), lr=lr, weight_decay=wd)
    elif opt_type == "KFAC":
        optimizer = KFACOptimizer(net.parameters(), lr=lr, weight_decay=wd)
    elif opt_type == "AggMo":
        optimizer = AggMo(net.parameters(), lr=lr, weight_decay=wd)
    elif opt_type == "AdaBelief":
        optimizer = AdaBelief(net.parameters(), lr=lr, weight_decay=wd)
    return optimizer

def rescale(net, net_base):
    for mod, mod_base in zip(net.modules(), net_base.modules()):
        if(isinstance(mod, nn.Conv2d)):
            #print( 'before='+ str( torch.norm(torch.norm(mod.weight, dim=(2,3), keepdim=True), dim=1, keepdim=True)[1]) )
            mod.weight.data = (mod.weight.data / torch.linalg.norm(mod.weight, dim=(1,2,3), keepdim=True)) * torch.linalg.norm(mod_base.weight, dim=(1,2,3), keepdim=True)
            #print( 'after='+ str( torch.norm(torch.norm(mod.weight, dim=(2,3), keepdim=True), dim=1, keepdim=True)[1]) )
    #print('end!')
    return net

def train(net, net_base, dataloader, optimizer, criterion, device, batch_size, epoch, ablate=False):

    start = time.time()
    net.train()

    correct = 0.0
    total = 0.0

    for batch_index, (images, labels) in enumerate(dataloader):

        labels = labels.to(device)
        images = images.to(device)

        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()
        total += batch_size

        if ablate:
            net = rescale(net, net_base)

        n_iter = (epoch - 1) * len(dataloader) + batch_index + 1

        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tAccuracy: {:0.4f}\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            correct.float() / total,
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index * batch_size + len(images),
            total_samples=len(dataloader.dataset)
        ), end="\r")

    finish = time.time()

    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))
    return correct.float() / len(dataloader.dataset)

@torch.no_grad()
def eval(net, dataloader, criterion, device, epoch=0, tb=True):

    start = time.time()
    net.eval()

    test_loss = 0.0 # cost function error
    correct = 0.0

    for (images, labels) in dataloader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = net(images)
        loss = criterion(outputs, labels)

        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    finish = time.time()

    print('Evaluating Network.....')
    print('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        epoch,
        test_loss / len(dataloader.dataset),
        correct.float() / len(dataloader.dataset),
        finish - start
    ))
    print()

    return correct.float() / len(dataloader.dataset)

def net_save(net, accs_dict, trained_root, suffix):
	print('Saving Model...')
	state = {
        'net': net.state_dict(),
        'Train_acc': accs_dict['Train'],
        'Test_acc': accs_dict['Test'],
        'lr': accs_dict['lr'],
    }

	torch.save(state, trained_root + 'VGG_model_{}.pth'.format(suffix))

	props_dict = {"params_list": net.params_list,
			  		"grads_list": net.grads_list,
					"activs_norms": [],
					"activs_corr": [],
					# "activs_ranks": [],
					"std_list": [],
					"grads_norms": [],
					}

	for mod in net.modules():
		if(isinstance(mod, Activs_prober)):
			props_dict["activs_norms"].append(mod.activs_norms)
			props_dict["activs_corr"].append(mod.activs_corr)
			# props_dict["activs_ranks"].append(mod.activs_ranks)
		if(isinstance(mod, Conv_prober)):
			props_dict["std_list"].append(mod.std_list)
			props_dict["grads_norms"].append(mod.grads_norms)

	print('Saving properties...')
	with open(trained_root + "properties_{}.pkl".format(suffix), 'wb') as f:
		pkl.dump(props_dict, f)
