from __future__ import annotations
import argparse
import os
import glob
import numpy as np
from typing import List, Tuple
from PIL import Image
from .features import compute_basic_features


def load_dataset(real_dir: str, ai_dir: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
	X: List[List[float]] = []
	y: List[int] = []
	feature_names: List[str] = []

	def add_dir(d: str, label: int):
		if not d:
			return
		for path in sorted(glob.glob(os.path.join(d, "**", "*"), recursive=True)):
			try:
				if not os.path.isfile(path):
					continue
				im = Image.open(path)
				im.verify()
				with open(path, "rb") as f:
					features, names = compute_basic_features(f.read())
					X.append(features)
					y.append(label)
					if not feature_names:
						feature_names = names
			except Exception:
				continue

	add_dir(real_dir, 0)
	add_dir(ai_dir, 1)

	if not X:
		raise RuntimeError("No training samples found.")

	Xn = np.array(X, dtype=np.float64)
	yn = np.array(y, dtype=np.float64)
	return Xn, yn, feature_names


def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, epochs: int = 400) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
	mean = X.mean(axis=0)
	std = X.std(axis=0)
	std = np.maximum(std, 1e-8)
	Xn = (X - mean) / std

	weights = np.zeros(Xn.shape[1], dtype=np.float64)
	bias = 0.0

	for _ in range(epochs):
		z = Xn @ weights + bias
		pred = 1.0 / (1.0 + np.exp(-z))
		error = pred - y
		grad_w = Xn.T @ error / Xn.shape[0]
		grad_b = float(np.sum(error) / Xn.shape[0])
		weights -= lr * grad_w
		bias -= lr * grad_b

	return weights, bias, mean, std


def main():
	parser = argparse.ArgumentParser(description="Train a simple logistic regression on basic image features.")
	parser.add_argument("--real_dir", type=str, required=True, help="Path to directory of real images")
	parser.add_argument("--ai_dir", type=str, required=True, help="Path to directory of AI-generated images")
	parser.add_argument("--out", type=str, default="model.npz", help="Output model path (npz)")
	parser.add_argument("--lr", type=float, default=0.1)
	parser.add_argument("--epochs", type=int, default=400)
	args = parser.parse_args()

	X, y, feature_names = load_dataset(args.real_dir, args.ai_dir)
	weights, bias, mean, std = train_logistic_regression(X, y, lr=args.lr, epochs=args.epochs)
	np.savez(args.out, weights=weights, bias=bias, mean=mean, std=std, feature_names=np.array(feature_names, dtype=object))
	print(f"Saved model to {args.out}  (features: {feature_names})")


if __name__ == "__main__":
	main()


