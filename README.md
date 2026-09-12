# Multi-Modal Geospatial AI Backend Gateway

A modular FastAPI backend featuring an Intent Classifier, LangGraph Agent State Machine & Task Planner, Extensible Tool Registry with remote model adapters (GeoLLaVA, RSICRC, Popeye, ResNet-50), Sih raster/fusion, and audit-grade execution traces.

This gateway does **not** load large model weights. Inference happens at configured remote HTTP endpoints.

---

## Key Features & Architecture

1. **FastAPI Gateway (`/api/v1`)**
   - `POST /api/v1/upload`: Upload files and receive `file_id` values. Upload **does not** run any model.
   - `POST /api/v1/query`: Classify intent, run the agent, call the required remote model, fuse evidence, return a trace.
   - `POST /api/v1/report`: Generate reports from execution trace IDs.
   - Thin model façades under `/api/v1/models/...` plus `GET /api/v1/models/health`.

2. **Natural Language Intent Classifier**
   - Routes to `VQA`, `CAPTIONING`, `GROUNDING`, `BI_TEMPORAL_CHANGE`, or `OPTICAL_SAR`.

3. **LangGraph Agent**
   - `Query` → classify → validate images → metadata → model dispatch → spatial analysis → Sih fusion → answer

4. **Remote model adapters**
   - **GeoLLaVA**: VQA, captioning, grounding (via `LLAVA_URL`). Sends a `.npy` array as multipart `file`.
   - **RSICRC**: bi-temporal change analysis (via `RSICRC_URL`). Sends RGB PNGs as multipart `before` + `after`.
   - **Popeye**: optical + SAR understanding (via `POPEYE_URL`). Sends RGB PNGs as multipart `optical_image` + `sar_image`.
   - **ResNet-50**: supporting features/domain service only (`RESNET_URL`). Not used on query routes.

---

## Query routing

| Intent | Tool | Remote model |
| --- | --- | --- |
| `VQA` | `VQA` | GeoLLaVA VQA |
| `CAPTIONING` | `Captioning` | GeoLLaVA caption |
| `GROUNDING` | `Grounding` | GeoLLaVA grounding |
| `BI_TEMPORAL_CHANGE` | `ChangeDetection` | RSICRC |
| `OPTICAL_SAR` | `OpticalSAR` | Popeye |

Qwen, YOLO, and RingMoGPT are **not** on live routes.

ResNet-50 is **not** invoked by ChangeDetection or `/query`. Use `POST /api/v1/models/resnet/features` only when you explicitly need features.

---

## Real inference vs mock vs misconfigured

Default: `MODEL_MOCK_MODE=false`.

| State | Condition | Behavior |
| --- | --- | --- |
| Real inference | mock flags false **and** URL set | HTTP POST to the remote wrapper |
| Explicit mock | `MODEL_MOCK_MODE=true` or a per-model `*_MOCK=true` | Deterministic stub with `"mock": true` |
| Misconfigured | mock false **and** URL empty/unreachable | **503** (missing/unavailable) or **504** (timeout). Never silent mock. |

`localhost` URLs mean that model is running on this PC (weights would be local to that process). Point URLs at a GPU host to keep weights off this machine.

Popeye is reached at `POPEYE_URL` (`POST /analyze` with multipart `query` + `optical_image` + `sar_image`). GeoLLaVA is reached at `LLAVA_URL` (`POST /vqa` with multipart `.npy` + question). RSICRC is reached at `RSICRC_URL` (`POST /analyze` with multipart `before` + `after` PNGs).

---

## Local vs remote

**Stays on this backend:** FastAPI, classifier, LangGraph, upload storage, Sih raster/preprocessing/fusion/reporting, thin HTTP adapters.

**Runs on a GPU host:** GeoLLaVA, RSICRC, Popeye, and (if used) ResNet-50 checkpoints. This repo does not contain `.pt` / `.pth` / `.safetensors` weights.

Point `RSICRC_URL` at the RSICRC Change Analysis API (FastAPI `/analyze`). The docs page may be `/docs`; this gateway strips that suffix automatically.

---

## Environment

Copy `.env.example` to `.env`. Placeholders only:

```
LLAVA_URL=
RSICRC_URL=
POPEYE_URL=
RESNET_URL=
MODEL_MOCK_MODE=false
LLAVA_MOCK=false
RSICRC_MOCK=false
POPEYE_MOCK=false
RESNET_MOCK=false
```

Do not commit secrets. Do not put API keys in `.env.example`.

---

## Model façades (this app) → remote wrappers

These routes belong to **this** FastAPI app. They resolve `image_id`s and POST to the configured URL.

| Local route | Remote |
| --- | --- |
| `POST /api/v1/models/llava/vqa` | `{LLAVA_URL}/vqa` (multipart `.npy`) |
| `POST /api/v1/models/llava/caption` | `{LLAVA_URL}/vqa` (multipart `.npy`) |
| `POST /api/v1/models/llava/grounding` | `{LLAVA_URL}/vqa` (multipart `.npy`) |
| `POST /api/v1/models/rsicrc/change` | `{RSICRC_URL}/analyze` (multipart `before` + `after` PNG) |
| `POST /api/v1/models/popeye/optical-sar` | `{POPEYE_URL}/analyze` (multipart PNG pair) |
| `POST /api/v1/models/resnet/features` | `{RESNET_URL}/features` |
| `GET /api/v1/models/health` | probes `/health` on each URL |

GeoLLaVA receives RGB as a `.npy` file over multipart form data. RSICRC receives RGB PNGs as multipart `before` and `after`.

---

## Image flow

```
POST /api/v1/upload
  → validate extension
  → store data/uploads/{file_id}_{original_filename}
  → return file_id
  → (no models)

POST /api/v1/query
  → classifier
  → BI_TEMPORAL_CHANGE: Sih load/validate/normalize → RGB PNG pair → RSICRC_URL `/analyze` → Sih fusion
  → VQA / caption / grounding: storage → RGB `.npy` → LLAVA_URL
  → OPTICAL_SAR: storage → RGB PNG pair → POPEYE_URL `/analyze`
```

---

## How to Run

### 1. Install Dependencies
```bash
python -m pip install -r requirements.txt
```

### 2. Configure `.env`
Set remote URLs for the models you have deployed. Leave unused URLs empty (those calls return 503 until configured).

### 3. Tests
```bash
python -m pytest tests/ -v
```

The suite mocks HTTP. It does not download weights or require a GPU.

### 4. Launch this gateway
```bash
uvicorn app.main:app --reload
```

Docs: `http://localhost:8000/docs`.

### 5. Point RSICRC at a GPU host

Set `RSICRC_URL` to the RSICRC Change Analysis API base URL (not this laptop, unless the GPU is here). Example:

```
RSICRC_URL=https://your-rsicrc-host.example
```

---

## Example query

```bash
curl -X POST http://localhost:8000/api/v1/upload -F "files=@t1.png" -F "files=@t2.png"
curl -X POST http://localhost:8000/api/v1/query -H "Content-Type: application/json" -d "{\"query\":\"What changed between these two images?\",\"image_ids\":[\"img-abc123\",\"img-def456\"]}"
```
