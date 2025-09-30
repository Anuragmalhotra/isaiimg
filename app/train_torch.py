from __future__ import annotations
import argparse
import os
from typing import Tuple, Optional, Dict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class TargetRemap:
	"""Picklable callable to remap class indices to Real=0, AI=1."""
	def __init__(self, remap: Dict[int, int]):
		self.remap = dict(remap)

	def __call__(self, y: int) -> int:
		return int(self.remap.get(int(y), int(y)))
from .torch_models import build_model


def _build_target_remap(ds: datasets.ImageFolder) -> Dict[int, int]:
	"""Ensure labels are consistent: Real -> 0, AI -> 1 regardless of folder order."""
	cls2idx = ds.class_to_idx
	ai_old = cls2idx.get('AI')
	real_old = cls2idx.get('Real')
	if ai_old is None or real_old is None:
		# Fallback: infer by name case-insensitively
		for name, idx in cls2idx.items():
			if name.lower() == 'ai':
				ai_old = idx
			elif name.lower() == 'real':
				real_old = idx
	if ai_old is None or real_old is None:
		# If still unknown, assume alphabetical mapping and that higher index is AI
		# But keep a stable binary mapping to 0/1
		classes_sorted = sorted(cls2idx.items(), key=lambda kv: kv[1])
		# map first seen to 0, second to 1
		return {classes_sorted[0][1]: 0, classes_sorted[1][1]: 1}
	return {real_old: 0, ai_old: 1}


def create_loaders(data_dir: str, img_size: int = 256, batch_size: int = 32, include_test: bool = True) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
	train_tfms = transforms.Compose([
		transforms.Resize((img_size, img_size)),
		transforms.RandomHorizontalFlip(),
		transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
		transforms.RandomRotation(10),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	])
	val_tfms = transforms.Compose([
		transforms.Resize((img_size, img_size)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	])
	train_root = os.path.join(data_dir, "train")
	val_root = os.path.join(data_dir, "val")
	train_ds_tmp = datasets.ImageFolder(train_root, transform=train_tfms)
	val_ds_tmp = datasets.ImageFolder(val_root, transform=val_tfms)
	remap = _build_target_remap(train_ds_tmp)
	target_tf = TargetRemap(remap)
	# Recreate datasets with target_transform applying the remap to 0/1
	train_ds = datasets.ImageFolder(train_root, transform=train_tfms, target_transform=target_tf)
	val_ds = datasets.ImageFolder(val_root, transform=val_tfms, target_transform=target_tf)
	# Use num_workers=0 to avoid multiprocessing pickling issues on some environments
	train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
	val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
	# Optional test loader if dataset provides test/ split
	test_loader: Optional[DataLoader] = None
	if include_test:
		test_root = os.path.join(data_dir, "test")
		if os.path.isdir(test_root):
			test_ds = datasets.ImageFolder(test_root, transform=val_tfms, target_transform=target_tf)
			test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
	return train_loader, val_loader, test_loader


def _compute_metrics(y_true: torch.Tensor, y_pred: torch.Tensor, y_score: Optional[torch.Tensor] = None) -> dict:
	"""Compute accuracy, precision, recall, F1 and confusion matrix for binary labels 0/1."""
	# Ensure tensors on CPU ints
	y_true = y_true.to("cpu").long()
	y_pred = y_pred.to("cpu").long()
	TP = int(((y_true == 1) & (y_pred == 1)).sum().item())
	TN = int(((y_true == 0) & (y_pred == 0)).sum().item())
	FP = int(((y_true == 0) & (y_pred == 1)).sum().item())
	FN = int(((y_true == 1) & (y_pred == 0)).sum().item())
	acc = (TP + TN) / max(1, (TP + TN + FP + FN))
	prec = TP / max(1, (TP + FP))
	rec = TP / max(1, (TP + FN))
	f1 = (2 * prec * rec / max(1e-12, (prec + rec))) if (prec + rec) > 0 else 0.0
	metrics = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "confusion_matrix": [[TN, FP], [FN, TP]]}
	if y_score is not None:
		# Save average score for AI class for reference
		metrics["avg_ai_score"] = float(y_score.to("cpu").mean().item())
	return metrics


def _evaluate(model: nn.Module, loader: DataLoader, device: str) -> dict:
	model.eval()
	all_true: list[int] = []
	all_pred: list[int] = []
	all_score: list[float] = []
	with torch.no_grad():
		for images, labels in loader:
			images = images.to(device)
			labels = labels.to(device)
			logits = model(images)
			scores = torch.softmax(logits, dim=1)[:, 1]  # P(class=1=AI)
			preds = logits.argmax(dim=1)
			all_true.append(labels)
			all_pred.append(preds)
			all_score.append(scores)
	# Concatenate
	y_true = torch.cat(all_true, dim=0)
	y_pred = torch.cat(all_pred, dim=0)
	y_score = torch.cat(all_score, dim=0)
	return _compute_metrics(y_true, y_pred, y_score)


def train(arch: str, data_dir: str, out: str, epochs: int = 5, lr: float = 3e-4, freeze_base: bool = True, device: str | None = None):
	device = device or ("cuda" if torch.cuda.is_available() else "cpu")
	train_loader, val_loader, test_loader = create_loaders(data_dir)
	model = build_model(arch=arch, num_classes=2, pretrained=True, freeze_base=freeze_base).to(device)
	# Class weights for imbalance: weight inversely proportional to class frequency in train set
	class_counts = [0, 0]
	for _, labels in train_loader:
		for y in labels:
			class_counts[int(y)] += 1
	weights = torch.tensor([
		1.0 / max(1, class_counts[0]),
		1.0 / max(1, class_counts[1])
	], dtype=torch.float32, device=device)
	criterion = nn.CrossEntropyLoss(weight=weights)
	optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

	best_acc = 0.0
	for epoch in range(epochs):
		model.train()
		running = 0.0
		correct = 0
		total = 0
		for images, labels in train_loader:
			images = images.to(device)
			labels = labels.to(device)
			optimizer.zero_grad()
			outputs = model(images)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()
			running += float(loss.item()) * images.size(0)
			preds = outputs.argmax(dim=1)
			correct += int((preds == labels).sum().item())
			total += int(labels.size(0))
		train_loss = running / max(1, total)
		train_acc = correct / max(1, total)

		# Evaluate on validation set with detailed metrics
		val_metrics = _evaluate(model, val_loader, device)
		val_acc = val_metrics["accuracy"]

		if val_acc > best_acc:
			best_acc = val_acc
			# Find best decision threshold on val (optimize F1) and save with checkpoint
			with torch.no_grad():
				# sweep thresholds
				best_t = 0.5
				best_f1 = -1.0
				# Collect scores/labels on val
				model.eval()
				all_true = []
				all_score = []
				for images, labels in val_loader:
					images = images.to(device)
					labels = labels.to(device)
					logits = model(images)
					scores = torch.softmax(logits, dim=1)[:, 1]
					all_true.append(labels.cpu())
					all_score.append(scores.cpu())
				y_true = torch.cat(all_true)
				y_score = torch.cat(all_score)
				for t in [i/100 for i in range(10, 91)]:
					pred = (y_score >= t).long()
					m = _compute_metrics(y_true, pred, y_score)
					if m["f1"] > best_f1:
						best_f1 = m["f1"]
						best_t = t
			torch.save({
				"arch": arch,
				"state_dict": model.state_dict(),
				"threshold": float(best_t),
			}, out)
			print(f"Epoch {epoch+1}: improved val_acc={val_acc:.4f} -> saved to {out}")
		else:
			print(f"Epoch {epoch+1}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f}")
		print(f"Val metrics: acc={val_metrics['accuracy']:.4f} prec={val_metrics['precision']:.4f} rec={val_metrics['recall']:.4f} f1={val_metrics['f1']:.4f} cm={val_metrics['confusion_matrix']}")

	print(f"Best val_acc={best_acc:.4f}. Model saved at {out}")

	# Optional test evaluation on unseen data if test/ exists
	if test_loader is not None:
		print("\nEvaluating on test split (unseen during training)...")
		test_metrics = _evaluate(model, test_loader, device)
		print(f"Test metrics: acc={test_metrics['accuracy']:.4f} prec={test_metrics['precision']:.4f} rec={test_metrics['recall']:.4f} f1={test_metrics['f1']:.4f} cm={test_metrics['confusion_matrix']}")


def main():
	parser = argparse.ArgumentParser(description="Train transfer learning classifier (AI vs Real)")
	parser.add_argument("--arch", type=str, default="resnet50", choices=["resnet50", "efficientnet_b0", "vit_b_16"])
	parser.add_argument("--data", type=str, required=True, help="Dataset root with train/ and val/ subfolders")
	parser.add_argument("--out", type=str, default="model.pt", help="Output checkpoint path")
	parser.add_argument("--epochs", type=int, default=5)
	parser.add_argument("--lr", type=float, default=3e-4)
	parser.add_argument("--unfreeze", action="store_true", help="Unfreeze base network for fine-tuning")
	parser.add_argument("--no-test", action="store_true", help="Skip evaluating on test/ even if present")
	args = parser.parse_args()

	# Temporarily toggle include_test via env var inside create_loaders; simplest to pass via global flag
	train(arch=args.arch, data_dir=args.data, out=args.out, epochs=args.epochs, lr=args.lr, freeze_base=not args.unfreeze)


if __name__ == "__main__":
	main()


