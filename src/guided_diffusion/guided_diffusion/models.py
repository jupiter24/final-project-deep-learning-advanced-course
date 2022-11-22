import torch
from torchvision import models
from .dist_util import dev


class DinoHead(torch.nn.Module):

	def __init__(self, in_dim=2048, out_dim=60000, hidden_dim=4096, bottleneck_dim=256):
		super().__init__()

		self.mlp = torch.nn.Sequential(*[
			torch.nn.Linear(in_dim, hidden_dim),
			torch.nn.BatchNorm1d(hidden_dim),
			torch.nn.GELU(),
			torch.nn.Linear(hidden_dim, bottleneck_dim)
		])
		self.last_layer = torch.nn.Linear(bottleneck_dim, out_dim, bias=False)

	def forward(self, x):
		x = self.mlp(x)
		return x


def get_dino_model():
	dino_model = torch.hub.load_state_dict_from_url(
		"https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain_full_checkpoint.pth",
		map_location=dev())['teacher']
	dino_model = {k.replace("backbone.", ""): v for k, v in dino_model.items()}

	resnet50 = models.resnet50()
	resnet50.fc = torch.nn.Identity()
	resnet50.head = DinoHead()
	resnet50.load_state_dict(dino_model, strict=True)
	resnet50.to(dev()).eval()

	return resnet50


def get_supervised_model():
	resnet50 = models.resnet50(pretrained=True)
	resnet50.fc = torch.nn.Identity()
	resnet50.to(dev()).eval()

	return resnet50
