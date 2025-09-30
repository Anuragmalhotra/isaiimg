from __future__ import annotations
from PIL import Image
import io
import math
from typing import List, Tuple


def _ensure_image(image_bytes: bytes) -> Image.Image:
	image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
	return image


def compute_basic_features(image_bytes: bytes) -> Tuple[List[float], List[str]]:
	image = _ensure_image(image_bytes)
	w, h = image.size

	# Resize to a manageable size for feature extraction consistency
	max_side = 256
	scale = min(1.0, max_side / max(w, h))
	if scale < 1.0:
		image = image.resize((max(1, int(w * scale)), max(1, int(h * scale))))
		w, h = image.size

	# 1) Global color stats
	pixels = image.load()
	count = w * h
	sum_r = 0.0
	sum_g = 0.0
	sum_b = 0.0
	sum_r2 = 0.0
	sum_g2 = 0.0
	sum_b2 = 0.0

	# 2) Edge density via simple Laplacian-like measure
	def get(px, py):
		px = min(max(px, 0), w - 1)
		py = min(max(py, 0), h - 1)
		return pixels[px, py]

	edge_sum = 0.0

	for y in range(h):
		for x in range(w):
			r, g, b = pixels[x, y]
			sum_r += r
			sum_g += g
			sum_b += b
			sum_r2 += r * r
			sum_g2 += g * g
			sum_b2 += b * b

			c = (r + g + b) / 3.0
			c_up = sum(get(x, y - 1)) / 3.0
			c_down = sum(get(x, y + 1)) / 3.0
			c_left = sum(get(x - 1, y)) / 3.0
			c_right = sum(get(x + 1, y)) / 3.0
			lap = abs(4 * c - (c_up + c_down + c_left + c_right))
			edge_sum += lap

	mean_r = sum_r / count
	mean_g = sum_g / count
	mean_b = sum_b / count
	var_r = sum_r2 / count - mean_r * mean_r
	var_g = sum_g2 / count - mean_g * mean_g
	var_b = sum_b2 / count - mean_b * mean_b
	std_r = max(0.0, var_r) ** 0.5
	std_g = max(0.0, var_g) ** 0.5
	std_b = max(0.0, var_b) ** 0.5

	edge_density = edge_sum / (count * 255.0)

	# 3) Color histogram entropy over 32 bins per channel
	bin_count = 32
	bin_size = 256 // bin_count
	hist_r = [0] * bin_count
	hist_g = [0] * bin_count
	hist_b = [0] * bin_count
	for y in range(h):
		for x in range(w):
			r, g, b = pixels[x, y]
			hist_r[min(r // bin_size, bin_count - 1)] += 1
			hist_g[min(g // bin_size, bin_count - 1)] += 1
			hist_b[min(b // bin_size, bin_count - 1)] += 1

	def entropy_of(hist):
		total = float(sum(hist))
		ent = 0.0
		for c in hist:
			if c <= 0:
				continue
			p = c / total
			ent -= p * math.log(p + 1e-12, 2)
		return ent

	ent_r = entropy_of(hist_r) / math.log(bin_count, 2)
	ent_g = entropy_of(hist_g) / math.log(bin_count, 2)
	ent_b = entropy_of(hist_b) / math.log(bin_count, 2)

	features = [
		mean_r / 255.0, mean_g / 255.0, mean_b / 255.0,
		std_r / 255.0, std_g / 255.0, std_b / 255.0,
		edge_density,
		ent_r, ent_g, ent_b,
	]
	feature_names = [
		"mean_r", "mean_g", "mean_b",
		"std_r", "std_g", "std_b",
		"edge_density",
		"entropy_r", "entropy_g", "entropy_b",
	]

	return features, feature_names


