from torchvision import models, transforms
import torchvision
import torch

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

	d_path = "./"
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
