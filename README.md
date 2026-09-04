# Multi-Modal Geospatial AI Backend Gateway

A modular FastAPI backend featuring an Intent Classifier, LangGraph Agent State Machine & Task Planner, Extensible Tool Registry with remote model adapters (GeoChat, CDChat, Popeye, ResNet-50), raster/fusion, and audit-grade execution traces.

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
   - `Query` → classify → validate images → metadata → model dispatch → spatial analysis → Rasterio fusion → answer

4. **Remote model adapters**
   - **GeoChat**: VQA, captioning, grounding (via `GEOCHAT_URL`)
   - **CDChat**: bi-temporal change description (via `CDCHAT_URL`)
   - **Popeye**: optical + SAR understanding (via `POPEYE_URL`)
   - **ResNet-50**: supporting features/domain service only (`RESNET_URL`). Not used on query routes.

---

## Query routing

| Intent | Tool | Remote model |
| --- | --- | --- |
| `VQA` | `VQA` | GeoChat VQA |
| `CAPTIONING` | `Captioning` | GeoChat caption |
| `GROUNDING` | `Grounding` | GeoChat grounding |
| `BI_TEMPORAL_CHANGE` | `ChangeDetection` | CDChat |
| `OPTICAL_SAR` | `OpticalSAR` | Popeye |

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

GeoChat and Popeye have **no official public HTTP API**. `GEOCHAT_URL` / `POPEYE_URL` are contracts for a self-hosted GPU wrapper that runs the actual model.

---

## Local vs remote

**Stays on this backend:** FastAPI, classifier, LangGraph, upload storage, raster/preprocessing/fusion/reporting, thin HTTP adapters.

**Runs on a GPU host:** GeoChat, CDChat, Popeye, and (if used) ResNet-50 checkpoints. This repo does not contain `.pt` / `.pth` / `.safetensors` weights.

CDChat wrapper (already in `services/cdchat/`):

```bash
uvicorn services.cdchat.main:app --host 0.0.0.0 --port 8001
```

Then set `CDCHAT_URL` to that host (not this laptop, unless the GPU is here).

---

## Environment

Copy `.env.example` to `.env`. Placeholders only:

```
GEOCHAT_URL=
CDCHAT_URL=
POPEYE_URL=
RESNET_URL=
MODEL_MOCK_MODE=false
GEOCHAT_MOCK=false
CDCHAT_MOCK=false
POPEYE_MOCK=false
RESNET_MOCK=false
```

Do not commit secrets. Do not put API keys in `.env.example`.

---

## Model façades (this app) → remote wrappers

These routes belong to **this** FastAPI app. They resolve `image_id`s and POST to the configured URL.

| Local route | Remote |
| --- | --- |
| `POST /api/v1/models/geochat/vqa` | `{GEOCHAT_URL}/vqa` |
| `POST /api/v1/models/geochat/caption` | `{GEOCHAT_URL}/caption` |
| `POST /api/v1/models/geochat/grounding` | `{GEOCHAT_URL}/grounding` |
| `POST /api/v1/models/cdchat/change` | `{CDCHAT_URL}/cdchat/predict` |
| `POST /api/v1/models/popeye/optical-sar` | `{POPEYE_URL}/optical-sar` |
| `POST /api/v1/models/resnet/features` | `{RESNET_URL}/features` |
| `GET /api/v1/models/health` | probes `/health` on each URL |

Remote wrapper contracts (ours, not native GeoChat/Popeye APIs) send RGB PNG as base64. CLIP 504px (GeoChat) and CDChat 448px BGR stay on the GPU service.

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
  → BI_TEMPORAL_CHANGE: Rasterio load/validate/normalize → RGB → CDCHAT_URL → Rasterio fusion
  → VQA / caption / grounding: storage → RGB → GEOCHAT_URL
  → OPTICAL_SAR: storage → RGB pair → POPEYE_URL
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

### 5. Launch CDChat on a GPU host (optional, separate process)

```bash
$env:CDCHAT_MODEL_PATH="C:\path\to\cdchat_lora"
$env:CDCHAT_MODEL_BASE="C:\path\to\llava-v1.5-7b"
$env:CDCHAT_MM_PROJECTOR_PATH="C:\path\to\mm_projector.bin"
$env:CDCHAT_DEVICE="cuda"
uvicorn services.cdchat.main:app --host 0.0.0.0 --port 8001
```

Set `CDCHAT_URL` on this backend to that host. `CDCHAT_SERVICE_MOCK` applies only to the CDChat **service** process, not this gateway.

---

## Example query

```bash
curl -X POST http://localhost:8000/api/v1/upload -F "files=@t1.png" -F "files=@t2.png"
curl -X POST http://localhost:8000/api/v1/query -H "Content-Type: application/json" -d "{\"query\":\"What changed between these two images?\",\"image_ids\":[\"img-abc123\",\"img-def456\"]}"
```
