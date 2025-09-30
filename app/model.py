from __future__ import annotations
import os
import numpy as np
from typing import Optional, Tuple, List, Dict, Any


class LocalModel:
	def __init__(self, weights: np.ndarray, bias: float, mean: np.ndarray, std: np.ndarray, feature_names: List[str]):
		self.weights = weights.astype(np.float64)
		self.bias = float(bias)
		self.mean = mean.astype(np.float64)
		self.std = np.maximum(std.astype(np.float64), 1e-8)
		self.feature_names = list(feature_names)

	def predict_proba_from_raw(self, raw_features: List[float]) -> float:
		x = np.array(raw_features, dtype=np.float64)
		x = (x - self.mean) / self.std
		z = float(np.dot(self.weights, x) + self.bias)
		return 1.0 / (1.0 + np.exp(-z))


def load_model_npz(path: str) -> Optional[LocalModel]:
	if not os.path.exists(path):
		return None
	data = np.load(path, allow_pickle=True)
	weights = data["weights"]
	bias = float(data["bias"]) if "bias" in data else 0.0
	mean = data["mean"] if "mean" in data else np.zeros_like(weights)
	std = data["std"] if "std" in data else np.ones_like(weights)
	feature_names = list(data["feature_names"]) if "feature_names" in data else [f"f{i}" for i in range(len(weights))]
	return LocalModel(weights=weights, bias=bias, mean=mean, std=std, feature_names=feature_names)


