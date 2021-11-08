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
import pickle as pkl

from models import Activs_prober, Conv_prober
from utils import progress_bar

def test(net, testloader, device, criterion):
	net.eval()
	test_loss = 0
	correct = 0
	total = 0
	with torch.no_grad():
		for batch_idx, (inputs, targets) in enumerate(testloader):
			inputs, targets = inputs.to(device), targets.to(device)
			outputs = net(inputs)
			loss = criterion(outputs, targets)
			test_loss += loss.item()
			_, predicted = outputs.max(1)
			total += targets.size(0)
			correct += predicted.eq(targets).sum().item()
			progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
				% (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
	return 100. * (correct / total)

def net_save(net, accs_dict, lr_warmup, trained_root, suffix, probe_layers=True):
	print('Saving Model...')
	state = {'net': net.state_dict(), 'Train_acc': accs_dict['Train'], 'Test_acc': accs_dict['Test']}
	if(lr_warmup):
		torch.save(state, trained_root + 'VGG_model_{}.pth'.format(suffix))
	else:
		torch.save(state, trained_root + 'VGG_model_{}.pth'.format(suffix))

		# torch.save(state, trained_root + '{layer}'.format(layer=args.norm_type) + '_conv_{layer}'.format(layer=args.conv_type) +
		# 	'_arch_{arch_name}'.format(arch_name=args.arch) + '_cfg_' + str(len(cfg_use)) + '_probed_' + args.probe_layers +
		# 	'_bsize_' + args.batch_size + '_init_lr_' + args.init_lr + '_skipinit_' + args.skipinit + '_grouping_' + args.p_grouping
		# 	+ '_seed_' + args.seed +'.pth')

	if(probe_layers):
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
