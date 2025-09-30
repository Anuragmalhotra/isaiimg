from __future__ import annotations
from PIL import Image
import io
import math

# Very naive heuristic-based "detector". For demo only.
# Returns a float in [0,1]: probability it's AI-generated.

def estimate_ai_probability(image_bytes: bytes) -> float:
	image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
	width, height = image.size

	# Heuristic 1: Over-smoothness (low edge density) can hint AI
	# Compute simple edge intensity using 3x3 Laplacian-like kernel on a downscaled image
	sample_max = 256
	scale = min(1.0, sample_max / max(width, height))
	if scale < 1.0:
		image_small = image.resize((max(1, int(width * scale)), max(1, int(height * scale))))
	else:
		image_small = image

	pixels = image_small.load()
	w, h = image_small.size

	def get(px, py):
		px = min(max(px, 0), w - 1)
		py = min(max(py, 0), h - 1)
		return pixels[px, py]

	edge_sum = 0.0
	for y in range(h):
		for x in range(w):
			r, g, b = get(x, y)
			c = (r + g + b) / 3.0
			c_up = sum(get(x, y - 1)) / 3.0
			c_down = sum(get(x, y + 1)) / 3.0
			c_left = sum(get(x - 1, y)) / 3.0
			c_right = sum(get(x + 1, y)) / 3.0
			lap = abs(4 * c - (c_up + c_down + c_left + c_right))
			edge_sum += lap

	edge_density = edge_sum / (w * h * 255.0)

	# Heuristic 2: Color entropy (low diversity may hint AI or compression)
	hist = image_small.histogram()
	total = sum(hist)
	entropy = 0.0
	for count in hist:
		if count <= 0:
			continue
		p = count / total
		entropy -= p * math.log(p + 1e-12, 2)

	# Normalize entropy roughly to [0,1] for RGB 8-bit
	entropy_norm = min(1.0, entropy / 16.0)

	# Combine heuristics into a score
	# Lower edge density and lower entropy push towards AI
	ai_score = (1.0 - min(edge_density * 4.0, 1.0)) * 0.6 + (1.0 - entropy_norm) * 0.4

	# Clip to [0,1]
	return max(0.0, min(1.0, ai_score))
