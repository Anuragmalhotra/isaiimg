# AI Image Detector (Demo)

A tiny FastAPI web app that estimates whether an image is AI-generated or real. By default it uses a simple local heuristic (demo only). If you provide an Illuminarty API key, it will call their classifier API.

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000` in your browser.

### Option A: Transfer learning with PyTorch (recommended)

Prepare a folder with `train/` and `val/` splits, each containing subfolders `real/` and `ai/`:

```
dataset/
  train/
    real/ ...
    ai/   ...
  val/
    real/ ...
    ai/   ...
```

Train a model (ResNet50 shown):

```bash
source .venv/bin/activate
python -m app.train_torch --arch resnet50 --data dataset --out model.pt --epochs 5
```

Run the server using the Torch model:

```bash
export TORCH_MODEL_PATH="model.pt"
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Supported `--arch`: `resnet50`, `efficientnet_b0`, `vit_b_16`.

### Option B: Classical local model (no Torch)

Prepare two folders of training images:

- `data/real/` for real photos
- `data/ai/` for AI-generated images

Train a simple logistic regression on handcrafted features:

```bash
source .venv/bin/activate
python -m app.train --real_dir data/real --ai_dir data/ai --out model.npz
```

Start the server so it loads your model automatically (reads `LOCAL_MODEL_PATH`, defaults to `model.npz` in project root):

```bash
export LOCAL_MODEL_PATH="model.npz"
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Notes
- Heuristic is not accurate. Prefer the external API or a trained model.
- Supports JPG/PNG/WebP etc. via Pillow.
- No persistence; files are processed in-memory.
