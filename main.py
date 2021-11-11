import os
import sys
import time
import math
import torch
import numpy as np
import argparse
import copy

from data import get_dataloader
from train import get_optimizer, train, LR_Scheduler, criterion
from test import test, net_save
from models import VGG

######### Parser #########
parser = argparse.ArgumentParser()
parser.add_argument("--probe_layers", help="Probe activations/gradients?", default='True', choices=['True', 'False'])
parser.add_argument("--cfg", help="Model configuration", default='cfg_10')
parser.add_argument("--dataset", help="CIFAR-10 or CIFAR-100", default='CIFAR-100', choices=['CIFAR-10', 'CIFAR-100'])
parser.add_argument("--batch_size", help="Batch size for DataLoader", default='256')
parser.add_argument("--opt_type", help="Optimizer", default='SGD', choices=['SGD'])
parser.add_argument("--seed", help="set random generator seed", default='0')
parser.add_argument("--download", help="download CIFAR-10/-100?", default='False')
args = parser.parse_args()

### temporary hard coded args ###
p_grouping = 32
probe_layers = True
ablate_bn=False
dataset = "CIFAR-100"
opt_type = "SGD"
batchsize = 256
seed = 0
download = True
expt = "full" # or "test"

trained_root = "trained/"
if torch.cuda.is_available():
	device='cuda'
	print ("CUDA available; using GPU")
else:
	device='cpu'


cfg = [64, (64, 2), 128, (128, 2), 256, (256, 2), 512, (512, 2), 512, 512]
lr = 1e-3
wd = 0.3
epochs = 100

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

				net = VGG(cfg, probe=probe_layers).to(device)
				net_base = copy.deepcopy(net).to(device)

				accs_dict = {'Train': [], 'Test': []}

				trainloader, testloader = get_dataloader(dataset, download, batchsize)
				optimizer = get_optimizer(net, opt_type=optimizer_type, lr=lr, wd=wd)

				stop_train = False
				for n in range(epochs):
					print('\nEpoch: {}'.format(n))
					train_acc, stop_train = train(net, net_base, trainloader, device, optimizer, criterion, ablation_setting)
					if(stop_train):
						break

					test_acc = test(net, testloader, device, criterion)
					accs_dict['Train'].append(train_acc)
					accs_dict['Test'].append(test_acc)

					if((batchsize==256 and n%5==0) or (batchsize<32)):
						net_save(net, accs_dict, lr, trained_root, suffix)
