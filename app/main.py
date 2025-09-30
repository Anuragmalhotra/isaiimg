from __future__ import annotations
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from .detector import estimate_ai_probability
from .features import compute_basic_features
from .model import load_model_npz, LocalModel
from typing import Optional
import os
import httpx
import asyncio
import io as _io
import time
import sys
import shutil
import subprocess

app = FastAPI(title="AI Image Detector", version="0.2.0")

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
def index_page():
	with open("static/index.html", "r", encoding="utf-8") as f:
		return f.read()

@app.get("/train", response_class=HTMLResponse)
def train_page():
	with open("static/train.html", "r", encoding="utf-8") as f:
		return f.read()

MODEL_PATH = os.getenv("LOCAL_MODEL_PATH", "model.npz")
_local_model: Optional[LocalModel] = load_model_npz(MODEL_PATH)

# Torch model (transfer learning) support
TORCH_MODEL_PATH = os.getenv("TORCH_MODEL_PATH", "model.pt")
_torch_model = None
_torch_arch: Optional[str] = None
_torch_threshold: float = 0.5
try:
	import torch
	from .torch_models import build_model
	if os.path.exists(TORCH_MODEL_PATH):
		ckpt = torch.load(TORCH_MODEL_PATH, map_location="cpu")
		_torch_arch = ckpt.get("arch", "resnet50")
		_torch_model = build_model(_torch_arch, num_classes=2, pretrained=False, freeze_base=False)
		_torch_model.load_state_dict(ckpt["state_dict"])
		_torch_model.eval()
		_torch_threshold = float(ckpt.get("threshold", 0.5))
except Exception:
	pass


@app.post("/api/detect")
async def detect(file: UploadFile = File(...)):
	image_bytes = await file.read()

	# Try Illuminarty API first if API key is configured
	api_key = os.getenv("ILLUMINARTY_API_KEY")
	if api_key:
		try:
			async with httpx.AsyncClient(timeout=20.0) as client:
				r = await client.post(
					"https://api.illuminarty.ai/v1/image/classify",
					headers={"X-API-Key": api_key},
					files={"file": (file.filename or "image", image_bytes, file.content_type or "application/octet-stream")},
				)
				r.raise_for_status()
				payload = r.json()

				# Attempt to extract a probability-like value; fall back to heuristic if missing
				prob = None
				for key in ("ai_probability", "aiProbability", "score", "ai_score", "probability"):
					if isinstance(payload, dict) and key in payload:
						try:
							prob = float(payload[key])
							break
						except Exception:
							pass

				if prob is not None:
					label = "AI" if prob >= 0.5 else "Real"
					return JSONResponse({
						"fileName": file.filename,
						"aiProbability": round(max(0.0, min(1.0, prob)), 3),
						"label": label,
						"provider": "illuminarty",
						"raw": payload,
					})
		except Exception:
			# fall back to local heuristic if external API fails
			pass

	# Torch transfer model (preferred if available)
	if _torch_model is not None:
		try:
			import torch
			from torchvision import transforms
			from PIL import Image
			img = Image.open(_io.BytesIO(image_bytes)).convert("RGB")
			prep = transforms.Compose([
				transforms.Resize((256, 256)),
				transforms.ToTensor(),
				transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
			])
			t = prep(img).unsqueeze(0)
			with torch.no_grad():
				logits = _torch_model(t)
				prob_ai = float(torch.softmax(logits, dim=1)[0, 1].item())
			return JSONResponse({
				"fileName": file.filename,
				"aiProbability": round(prob_ai, 3),
				"label": "AI" if prob_ai >= _torch_threshold else "Real",
				"provider": f"torch:{_torch_arch or 'resnet50'}",
				"threshold": _torch_threshold,
			})
		except Exception:
			pass

	# Local classic model (if available)
	if _local_model is not None:
		features, _ = compute_basic_features(image_bytes)
		score = float(_local_model.predict_proba_from_raw(features))
		return JSONResponse({
			"fileName": file.filename,
			"aiProbability": round(score, 3),
			"label": "AI" if score >= 0.5 else "Real",
			"provider": "local-model"
		})

	# Fallback: heuristic (when no model yet)
	score = estimate_ai_probability(image_bytes)
	return JSONResponse({
		"fileName": file.filename,
		"aiProbability": round(score, 3),
		"label": "AI" if score >= 0.5 else "Real",
		"provider": "heuristic"
	})


# Feedback API: accept corrected labels and store for retraining
@app.post("/api/feedback")
async def feedback(label: str = Form(...), file: UploadFile = File(...)):
	label = (label or "").strip().upper()
	if label not in ("AI", "REAL"):
		return JSONResponse({"ok": False, "error": "label must be AI or Real"}, status_code=400)
	buf = await file.read()
	os.makedirs("data/feedback/AI", exist_ok=True)
	os.makedirs("data/feedback/Real", exist_ok=True)
	stamp = int(time.time() * 1000)
	name = file.filename or f"image_{stamp}.jpg"
	# sanitize filename
	name = name.replace("/", "_")
	out_dir = f"data/feedback/{'AI' if label=='AI' else 'Real'}"
	out_path = os.path.join(out_dir, f"{stamp}_{name}")
	with open(out_path, "wb") as f:
		f.write(buf)
	return {"ok": True, "saved": out_path}


def _create_splits_from_data_and_feedback(project_root: str) -> str:
	"""Combine data/ and data/feedback/ into data_splits/ with 80/10/10 split."""
	root_src = os.path.join(project_root, "data")
	root_fb = os.path.join(project_root, "data", "feedback")
	root_dst = os.path.join(project_root, "data_splits")
	# Rebuild destination
	if os.path.isdir(root_dst):
		shutil.rmtree(root_dst)
	for split in ("train", "val", "test"):
		for cls in ("AI", "Real"):
			os.makedirs(os.path.join(root_dst, split, cls), exist_ok=True)
	def list_images(d: str):
		if not os.path.isdir(d):
			return []
		allowed = {".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp"}
		out = []
		for nm in os.listdir(d):
			p = os.path.join(d, nm)
			if os.path.isfile(p) and os.path.splitext(p)[1].lower() in allowed:
				out.append(p)
		return out
	import random
	random.seed(42)
	for cls in ("AI", "Real"):
		src_cls = os.path.join(root_src, cls)
		fb_cls = os.path.join(root_fb, cls)
		files = list_images(src_cls) + list_images(fb_cls)
		random.shuffle(files)
		n = len(files)
		n_train = int(0.8 * n)
		n_val = max(1, int(0.1 * n)) if n >= 10 else max(0, int(0.1 * n))
		splits = {
			"train": files[:n_train],
			"val": files[n_train:n_train+n_val],
			"test": files[n_train+n_val:],
		}
		# ensure at least 1 in val if possible
		if len(splits["val"]) == 0 and len(splits["train"]) > 1:
			splits["val"].append(splits["train"].pop())
		for split, flist in splits.items():
			for src_path in flist:
				base = os.path.basename(src_path)
				dst = os.path.join(root_dst, split, cls, base)
				try:
					shutil.copy2(src_path, dst)
				except Exception:
					pass
	return root_dst


_train_task: Optional[asyncio.Task] = None
_train_log_path: Optional[str] = None
_train_started_at: Optional[float] = None


@app.post("/api/train")
async def train_now(arch: str = Form("efficientnet_b0"), epochs: int = Form(8), unfreeze: bool = Form(True)):
	global _train_task
	if _train_task and not _train_task.done():
		return {"ok": False, "status": "training_already_running"}

	project_root = os.getcwd()
	data_dir = _create_splits_from_data_and_feedback(project_root)
	python_bin = sys.executable
	out_path = os.path.join(project_root, "model.pt")

	async def _runner():
		cmd = [python_bin, "-u", "-m", "app.train_torch", "--arch", arch, "--data", data_dir, "--out", out_path, "--epochs", str(int(epochs))]
		if bool(unfreeze):
			cmd.append("--unfreeze")
		# log stdout/stderr to file
		global _train_log_path, _train_started_at
		_train_log_path = os.path.join(project_root, "train.out")
		_train_started_at = time.time()
		log_f = open(_train_log_path, "w", buffering=1)
		try:
			proc = await asyncio.create_subprocess_exec(*cmd, cwd=project_root, stdout=log_f, stderr=log_f)
			await proc.wait()
		finally:
			try:
				log_f.flush(); log_f.close()
			except Exception:
				pass
		# Reload model in-process
		try:
			import torch as _torch
			from .torch_models import build_model as _build
			if os.path.exists(out_path):
				ckpt = _torch.load(out_path, map_location="cpu")
				arch2 = ckpt.get("arch", arch)
				model2 = _build(arch2, num_classes=2, pretrained=False, freeze_base=False)
				model2.load_state_dict(ckpt["state_dict"])
				model2.eval()
				threshold2 = float(ckpt.get("threshold", 0.5))
				globals()["_torch_model"] = model2
				globals()["_torch_arch"] = arch2
				globals()["_torch_threshold"] = threshold2
		except Exception:
			pass

	_train_task = asyncio.create_task(_runner())
	return {"ok": True, "status": "started"}


@app.get("/api/train_status")
async def train_status():
	running = bool(_train_task and not _train_task.done())
	lines: list[str] = []
	if _train_log_path and os.path.exists(_train_log_path):
		try:
			with open(_train_log_path, "r", encoding="utf-8", errors="ignore") as f:
				content = f.read()
				lines = content.strip().splitlines()[-50:]
		except Exception:
			lines = []
	return {
		"ok": True,
		"running": running,
		"startedAt": _train_started_at,
		"logTail": lines,
	}
