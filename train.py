import os
import sys
import time
import math
import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from utils import progress_bar


base_epochs = [40, 20] # Number of epochs to train for
wd = 1e-4
total_epochs = np.sum(base_epochs)

def train_network(dataset, bsize):
    base_sched = [1e-1 * 256 / bsize, 1e-2 * 256 / bsize] # Learning rate is linearly scaled according to batch-size
    if (bsize<32):
    	base_epochs = [8, 2] # 10 epochs at batch-size of 16 have 2x number of iterations as 60 epochs of batch-size 256 training


######### Loss #########
criterion = nn.CrossEntropyLoss()

######### Optimizers #########
def get_optimizer(net, lr, wd, opt_type="SGD"):
	if(opt_type=="SGD"):
		optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
	return optimizer

class LR_Scheduler(object):
    def __init__(self, optimizer, warmup_epochs, warmup_lr, num_epochs, base_lr, final_lr, iter_per_epoch):
        self.base_lr = base_lr
        warmup_iter = iter_per_epoch * warmup_epochs
        warmup_lr_schedule = np.linspace(warmup_lr, base_lr, warmup_iter)
        decay_iter = iter_per_epoch * (num_epochs - warmup_epochs)
        lr_sched = np.array([base_lr] * int(decay_iter * 0.75) + [final_lr] * int(decay_iter * 0.25))
        self.lr_schedule = np.concatenate((warmup_lr_schedule, lr_sched))
        self.optimizer = optimizer
        self.iter = 0
        self.current_lr = 0

    def step(self):
        for param_group in self.optimizer.param_groups:
            lr = param_group['lr'] = self.lr_schedule[self.iter]

        self.iter += 1
        self.current_lr = lr
        return lr
    def get_lr(self):
        return self.current_lr

######### Training functions #########
# Training
def train(net, trainloader, device, optimizer, criterion, scheduler):
	net.train()
	train_loss = 0
	correct = 0
	total = 0
	stop_train = False
	for batch_idx, (inputs, targets) in enumerate(trainloader):
		inputs, targets = inputs.to(device), targets.to(device)
		optimizer.zero_grad()
		outputs = net(inputs)
		loss = criterion(outputs, targets)
		loss.backward()
		optimizer.step()
		scheduler.step()
		train_loss += loss.item()
		_, predicted = outputs.max(1)
		total += targets.size(0)
		correct += predicted.eq(targets).sum().item()
		if(np.isnan(train_loss)):
			stop_train=True
			break
		progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%.5f)'
			% (train_loss/(batch_idx+1), 100.*correct/total, correct, optimizer.param_groups[0]['lr']))
	return 100. * (correct / total), stop_train
