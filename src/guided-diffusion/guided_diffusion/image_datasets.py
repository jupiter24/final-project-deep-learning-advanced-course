import torch as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt


def load_data(batch_size=50, shuffle=True, num_workers=1, pin_memory=True):
	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	transform = transforms.Compose([
		transforms.RandomResizedCrop(224),  # they do it centered
		transforms.RandomHorizontalFlip(),
		# if we flip it before getting the representation then we have to do it for the diffusin model too
		transforms.ToTensor(),
		normalize,  # seems that they don't do it at least for the SSL
	])

	train_dataset = datasets.ImageFolder(
		root='../../../data/train_2classes',
		transform=transform
	)

	# training data loaders
	train_loader = DataLoader(
		train_dataset, batch_size=batch_size, shuffle=shuffle,
		num_workers=num_workers, pin_memory=pin_memory
	)

	while True:
		yield from train_loader
