# OCR Chinese Text Masking (PDF)

Pipeline for extracting binary masks of Chinese text in scanned engineering drawings.

## What it does

- Renders PDF pages to grayscale images with optional CLAHE.
- Detects oriented text proposals (PaddleOCR backend, MSER fallback).
- Filters text/non-text proposals (heuristics or trained classifier).
- Rasterizes polygons into binary page masks (0/255).
- Produces per-page overlays, proposals, and metrics reports.

## Install

### Linux (recommended / primary)

#### System packages

Poppler (for `pdftoppm`) and common runtime libs for OpenCV wheels:

```bash
sudo apt update
sudo apt install -y poppler-utils libgl1 libglib2.0-0
```

Optional (if you want to build some Python deps from source):

```bash
sudo apt install -y build-essential python3-dev
```

#### Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip wheel setuptools
pip install -e .
```

Optional detector extras:

```bash
pip install -e ".[detector]"
```

#### Render backend (Poppler vs PyMuPDF)

- **Default**: `--render-backend auto` (tries best available)
- **Poppler**: on Linux it’s usually on PATH already once `poppler-utils` is installed.

Examples:

```bash
maskpdf run "scheme.pdf" --out "output" --dpi 400 --render-backend poppler
```

If `pdftoppm` is not on PATH, provide it explicitly:

```bash
maskpdf run "scheme.pdf" --out "output" --dpi 400 --render-backend poppler --poppler-path "/usr/bin"
```

#### OCR backends

The project supports:

- **RapidOCR (ONNX Runtime)**: easiest to install; used by default in many setups
- **PaddleOCR**: can be used for recognition/detection when installed
- **Paddle bridge mode**: if PaddleOCR runtime is not importable in the current env, you can run it via a separate python interpreter

Bridge env vars (optional):

```bash
export OCR_PADDLE_PYTHON="/path/to/paddle/python"
export OCR_PADDLE_BRIDGE_SCRIPT="/absolute/path/to/scripts/paddle_ocr_bridge.py"
```

Notes:

- If PaddleOCR is unavailable, the app will fall back to RapidOCR automatically for region text extraction.
- For best speed, keep `mode=fast` (default) unless you explicitly need `accurate`.

### Windows (quick)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

Optional detector dependency:

```bash
pip install -e .[detector]
```

## Run inference

```bash
maskpdf run "scheme.pdf" --out "output" --dpi 400
```

Use explicit Poppler backend and local `pdftoppm` folder:

```bash
maskpdf run "scheme.pdf" --out "output" --dpi 400 --render-backend poppler --poppler-path "C:\Users\Valentin\Downloads\sollers-translate-pdf\third_party\poppler\poppler-25.12.0\Library\bin"
```

With ground-truth masks for evaluation:

```bash
maskpdf run "scheme.pdf" --out "output" --dpi 400 --gt-masks "data/gt_masks"
```

Outputs:

- `output/rendered_pages/page_XXXX.png`
- `output/proposals/page_XXXX_proposals.json`
- `output/masks/page_XXXX_mask.png`
- `output/region_crops/page_XXXX/page_XXXX_YYYYY.png` (each mask zone saved separately)
- `output/overlays/page_XXXX_overlay.png`
- `output/report.json`

## Run local Web UI

Start server:

```bash
maskpdf web --host 127.0.0.1 --port 8000
```

With explicit Poppler:

```bash
maskpdf web --host 127.0.0.1 --port 8000 --render-backend poppler --poppler-path "C:\Users\Valentin\Downloads\sollers-translate-pdf\third_party\poppler\poppler-25.12.0\Library\bin"
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000), upload PDF, click **Generate Mask + OCR**, and click any highlighted region to open copyable text.

Note: if PaddleOCR runtime is unavailable (for example `paddle` is not installed), the app automatically falls back to `RapidOCR` for region text extraction.

## Labeling setup

Create manifest + split:

```bash
maskpdf prepare-labeling --rendered-dir output/rendered_pages --manifest data/manifest.json --dev-ratio 0.5 --seed 42
```

Create helper list for CVAT / Label Studio:

```bash
maskpdf prepare-labeling --rendered-dir output/rendered_pages --manifest data/manifest.json --cvat-index-dir data/cvat
```

Annotation format in manifest (`annotations` per page):

```json
{
  "polygon": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
  "label": "text"
}
```

## Filter training & calibration

1) Build feature dataset from proposals + GT masks:

```bash
python scripts/build_filter_features.py \
  --proposals-dir output/proposals \
  --gt-masks-dir data/gt_masks \
  --rendered-dir output/rendered_pages \
  --out data/filter_features.json
```

2) Train classifier:

```bash
maskpdf train-filter --features data/filter_features.json --model-out models/filter.joblib --trees 300
```

3) Calibrate threshold (maximize recall with low FPR):

```bash
maskpdf calibrate --scores-json data/filter_scores.json --target-recall 1.0
```

4) Write detector fine-tune recipe:

```bash
maskpdf write-finetune-recipe --dataset-root data --out configs/paddle_finetune_recipe.json
```

## Notes on target metrics

- `100% recall` and `<5% FP` usually require iterative hard-negative mining.
- Keep detector thresholds permissive first (maximize recall), then reduce FP with classifier calibration.
- Use stratified dev/test splits covering different drawing styles and text sizes.

## Web API smoke test

```bash
python scripts/smoke_web_api.py --pdf scheme.pdf --poppler-path "C:\Users\Valentin\Downloads\sollers-translate-pdf\third_party\poppler\poppler-25.12.0\Library\bin"
```

## OCR quality comparison (regions)

Evaluate current recognition quality from `regions/*.json`:

```bash
python scripts/evaluate_regions.py --regions-dir output_poppler/regions
```

Compare before/after directories:

```bash
python scripts/evaluate_regions.py --regions-dir output_after/regions --baseline-regions-dir output_before/regions
```
