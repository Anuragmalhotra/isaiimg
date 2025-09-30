from __future__ import annotations
import torch
import torch.nn as nn
from torchvision import models


def build_model(arch: str, num_classes: int = 2, pretrained: bool = True, freeze_base: bool = True) -> nn.Module:
	arch = arch.lower()
	if arch == "resnet50":
		m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
		in_feats = m.fc.in_features
		m.fc = nn.Linear(in_feats, num_classes)
	elif arch == "efficientnet_b0":
		m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
		in_feats = m.classifier[-1].in_features
		m.classifier[-1] = nn.Linear(in_feats, num_classes)
	elif arch in ("vit_b_16", "vit_b16", "vit"):
		m = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None)
		in_feats = m.heads.head.in_features
		m.heads.head = nn.Linear(in_feats, num_classes)
	else:
		raise ValueError(f"Unsupported arch: {arch}")

	if freeze_base:
		for name, param in m.named_parameters():
			param.requires_grad = False
		# unfreeze classifier head
		for p in m.parameters():
			pass
		# find classifier parameters and unfreeze them
		if hasattr(m, "fc"):
			for p in m.fc.parameters():
				p.requires_grad = True
		elif hasattr(m, "classifier"):
			for p in m.classifier.parameters():
				p.requires_grad = True
		elif hasattr(m, "heads") and hasattr(m.heads, "head"):
			for p in m.heads.head.parameters():
				p.requires_grad = True

	return m


