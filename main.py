import os
import sys
import time
import math
import torch
import numpy as np
import argparse

from data import get_dataloader
from train import get_optimizer, train, LR_Scheduler, criterion
from test import test, net_save
from models import VGG
from utils import progress_bar

######### Parser #########
parser = argparse.ArgumentParser()
parser.add_argument("--p_grouping", help="Number of channels per group for GroupNorm", default='32', choices=['1', '0.5', '0.25', '0.125', '0.0625', '0.03125', '0.0000001', '8', '16', '32', '64'])
parser.add_argument("--probe_layers", help="Probe activations/gradients?", default='True', choices=['True', 'False'])
parser.add_argument("--cfg", help="Model configuration", default='cfg_10')
parser.add_argument("--skipinit", help="Use skipinit initialization?", default='False', choices=['True', 'False'])
parser.add_argument("--preact", help="Use preactivation variants for ResNet?", default='False', choices=['True', 'False'])
parser.add_argument("--dataset", help="CIFAR-10 or CIFAR-100", default='CIFAR-100', choices=['CIFAR-10', 'CIFAR-100'])
parser.add_argument("--batch_size", help="Batch size for DataLoader", default='256')
parser.add_argument("--init_lr", help="Initial learning rate", default='1')
parser.add_argument("--lr_warmup", help="Use a learning rate warmup?", default='False', choices=['True', 'False'])
parser.add_argument("--opt_type", help="Optimizer", default='SGD', choices=['SGD'])
parser.add_argument("--seed", help="set random generator seed", default='0')
parser.add_argument("--download", help="download CIFAR-10/-100?", default='False')
args = parser.parse_args()

### temporary hard coded args ###
p_grouping = 32
probe_layers = True
ablate_bn = False # remove adaptive optimization that arises from BN
skipinit = False
preact = False
dataset = "CIFAR-100"
init_lr = 1
opt_type = "SGD"
batchsize = 256
seed = 0
download = True
lr_warmup = False

trained_root = "trained/"
if torch.cuda.is_available():
	device='cuda'
	print ("CUDA available; using GPU")
else:
	device='cpu'
expt = "full" # or "test"

cfg = [64, (64, 2), 128, (128, 2), 256, (256, 2), 512, (512, 2), 512, 512]

base_sched_iter = [1e-1, 1e-2] # LR Schedule
base_epochs_iter = [40, 20] # Number of epochs to train for
wd_base = 1e-4

base_sched, base_epochs, wd = base_sched_iter, base_epochs_iter, wd_base # Training configuration
total_epochs = np.sum(base_epochs)

if __name__ == "__main__":
	if not os.path.isdir("trained"):
		os.mkdir("trained")
	if not os.path.isdir("datasets"):
		os.mkdir("datasets")

	if expt == "full":
		for ablation_setting in [False, True]:
			# for optimizer_type in ["SGD", "RMSProp", "Adam"]:
			for optimizer_type in ["Adam", "SGD"]:
				suffix = "ablate{}_optim{}".format(str(ablation_setting), optimizer_type)

				# Train
				print("\n------------------ Training ------------------\n")
				print("Training: Ablate {}, Optimizer {}".format(ablation_setting, optimizer_type))
				best_acc = 0
				lr_ind = 0
				epoch = 0
				base_lr = init_lr * base_sched[0]
				final_lr = init_lr * base_sched[-1]

				if(lr_warmup):
					warmup_lr = 0
					warmup_epochs = 1 if batchsize==16 else 5
				else:
					warmup_lr = base_lr
					warmup_epochs = 0

				net = VGG(cfg, probe=False).to(device)
				accs_dict = {'Train': [], 'Test': []}

				trainloader, testloader = get_dataloader(dataset, download, batchsize)
				optimizer = get_optimizer(net, opt_type=optimizer_type, lr=warmup_lr, wd=wd, ablate_bn=ablation_setting)
				scheduler = LR_Scheduler(optimizer, warmup_epochs=warmup_epochs, warmup_lr=warmup_lr, num_epochs=total_epochs, base_lr=base_lr, final_lr=final_lr, iter_per_epoch=len(trainloader))

				stop_train = False
				while(lr_ind < len(base_sched)):
					if(stop_train):
						break
					print("\n--learning rate is {}".format(optimizer.param_groups[0]['lr']))
					for n in range(base_epochs[lr_ind]):
						print('\nEpoch: {}'.format(epoch))
						train_acc, stop_train = train(net, trainloader, device, optimizer, criterion, scheduler)
						if(stop_train):
							break
						test_acc = test(net, testloader, device, criterion)
						accs_dict['Train'].append(train_acc)
						accs_dict['Test'].append(test_acc)
						epoch += 1
						if((batchsize==256 and epoch%5==0) or (batchsize<32)):
							net_save(net, accs_dict, lr_warmup, trained_root, suffix)
					lr_ind += 1

	else:
		suffix = "basic"
		# Train
		print("\n------------------ Training ------------------\n")
		best_acc = 0
		lr_ind = 0
		epoch = 0
		base_lr = init_lr * base_sched[0]
		final_lr = init_lr * base_sched[-1]

		if(lr_warmup):
			warmup_lr = 0
			warmup_epochs = 1 if batchsize==16 else 5
		else:
			warmup_lr = base_lr
			warmup_epochs = 0

		net = VGG(cfg).to(device)
		accs_dict = {'Train': [], 'Test': []}

		trainloader, testloader = get_dataloader(dataset, download, batchsize)
		optimizer = get_optimizer(net, opt_type=opt_type, lr=warmup_lr, wd=wd, ablate_bn=ablate_bn)
		scheduler = LR_Scheduler(optimizer, warmup_epochs=warmup_epochs, warmup_lr=warmup_lr, num_epochs=total_epochs, base_lr=base_lr, final_lr=final_lr, iter_per_epoch=len(trainloader))

		stop_train = False
		while(lr_ind < len(base_sched)):
			if(stop_train):
				break
			print("\n--learning rate is {}".format(optimizer.param_groups[0]['lr']))
			for n in range(base_epochs[lr_ind]):
				print('\nEpoch: {}'.format(epoch))
				train_acc, stop_train = train(net, trainloader, device, optimizer, criterion, scheduler)
				if(stop_train):
					break
				test_acc = test(net, testloader, device, criterion)
				accs_dict['Train'].append(train_acc)
				accs_dict['Test'].append(test_acc)
				epoch += 1
				if((batchsize==256 and epoch%5==0) or (batchsize<32)):
					net_save(net, accs_dict, lr_warmup, trained_root, suffix)
			lr_ind += 1
