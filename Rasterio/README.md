# Satellite Imagery Analysis — Backend Data & Verification Layer

An AI-powered satellite imagery analysis system that lets a user ask
natural-language questions about Earth-observation images and receive an
explainable, evidence-backed answer. The system takes GeoTIFF/TIFF imagery as
input and uses an agentic backend to understand the query, decide which
analysis is required, and route the request to the appropriate specialist AI
model — single-image visual question answering and scene captioning,
text-guided region grounding, bi-temporal change detection and change
description, or optical-SAR analysis.

The overall flow is:

```
User Query → Agent/Intent Routing → Specialist AI Models → Evidence Fusion
          → Verified Result → Storage/PDF Report → Interactive Dashboard
```

The goal is not merely to produce an AI prediction, but to provide a
multi-modal, explainable, auditable satellite-image analysis system.

## This module (Member 5 — backend data & verification layer)

This repository is the backend data and verification layer:

- **Raster handling** — metadata and band extraction with Rasterio/GDAL/OpenCV,
  image preprocessing, and temporal (bi-temporal) data handling.
- **Standardized model I/O** — common input/output interfaces between the raster
  pipeline and the specialist AI models.
- **Multi-Evidence Fusion Engine** — combines semantic, temporal, spatial, and
  optical-SAR evidence into a final confidence score and decision.
- **Storage & reporting** — stores analysis results as JSON and generates an
  automated PDF report with ReportLab.

## Project structure

```
raster/          Raster I/O and preprocessing
  bands.py         Load and stack the 13 Sentinel-2 bands
  temporal.py      Load a bi-temporal image pair for a sample
  dataset.py       Discover and validate OSCD dataset samples
  metadata.py      Extract GeoTIFF metadata
  alignment.py     Pixel-difference alignment check
  preprocessing.py Band normalization (min-max)
  model_input.py   ModelInput dataclass + preprocessing
  model_output.py  ModelOutput dataclass
fusion/
  evidence.py      Evidence + weighted fusion engine
storage/
  results.py       Save/load results as JSON
reports/
  generator.py     Generate a PDF report from a stored result
data/            OSCD (Onera Satellite Change Detection) dataset
results/         Stored analysis results (generated)
reports_output/  Generated PDF reports
```

## Setup

Requires Python 3.10+ and the GDAL system library (a dependency of Rasterio).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Each module is runnable directly as a self-test (from the repository root):

```bash
python -m raster.dataset        # validate every dataset sample
python -m raster.temporal       # load one temporal pair
python -m raster.model_input    # build a standardized model input
python -m fusion.evidence       # run the evidence fusion demo
python -m storage.results       # save/load a result
python -m reports.generator     # generate a PDF report for train_000
```

## Dataset

Uses the **Onera Satellite Change Detection (OSCD)** dataset. Each sample
contains two temporal acquisitions (`imgs_1_rect/` and `imgs_2_rect/`), each
with 13 Sentinel-2 bands (`B01`–`B12`, `B8A`) as separate GeoTIFFs.
