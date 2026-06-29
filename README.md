---
title: Swiss Recycling Assistant
emoji: ♻️
colorFrom: blue
colorTo: green
sdk: docker
app_file: app.py
pinned: false
---

# Swiss Waste Recycling Assistant

Bachelor Thesis, ZHAW Life Sciences und Facility Management  
Institut für Computational Life Sciences, Wädenswil  
Author: Lejla Beganovic | Supervisor: Martin Schüle

## Overview

An AI-powered assistant for Swiss waste disposal guidance. The system combines a
fine-tuned MobileNetV3 image classifier with a LangGraph-based GPT-4o agent that
provides disposal instructions strictly aligned with Swiss Recycle guidelines.

Users can upload a photo of a waste item, ask disposal questions in text, or enter
their municipality to find nearby collection points on an interactive map. The
interface supports German and English.

## Architecture

### Agent Pipeline

The agent is implemented as a LangGraph `StateGraph` with six nodes:

| Node | Role |
|---|---|
| Perception | Detects input type: image upload, location query, or text question |
| Classifier | Runs the MobileNetV3 classifier on uploaded photos |
| Knowledge Base | Maps the category or text keywords to Swiss Recycle disposal guidelines |
| Geolocation | Geocodes the municipality, queries OSM collection points, returns coordinates |
| Response | Builds the LLM prompt from retrieved context and calls GPT-4o |
| Clarification | Returns a targeted follow-up question when classifier confidence is too low |

Routing is conditional: image inputs go through Classifier before Knowledge Base;
location queries bypass Knowledge Base and go directly to Geolocation; text queries
go directly to Knowledge Base. The Clarification node short-circuits the Response
node when the model is uncertain.

### LLM

OpenAI GPT-4o via `langchain-openai`. An earlier approach using a local Ollama LLM
was evaluated and discarded in favour of GPT-4o for reliability and output quality.

### Image Classifier

- Architecture: MobileNetV3-Large, pre-trained on ImageNet (IMAGENET1K\_V2 weights)
- Fine-tuned classification head for 17 Swiss waste categories
- Approximately 4,866,625 parameters
- Hosted on Hugging Face Hub at `le7lum/swiss-waste-classifier`, downloaded at
  runtime via `hf_hub_download`

### Map and Collection Points

Collection points are displayed on a Leaflet map embedded in the dashboard as an
iframe served by a Flask route. Municipality names and ZIP codes are geocoded via
the geo.admin.ch API. Collection point data (recycling containers, waste disposal
centres) is queried from OpenStreetMap via the Overpass API.

## Dataset

- Source: Waste image dataset from Helene Benkert (Master's thesis "Don't Waste
  Compute: Efficient AI Models for Waste Management")
- Split: stratified random split
- Train: 4,336 images | Validation: 929 images | Test: 930 images | Total: 6,195 images
- 17 classes:

```
aluminium, brown_glass, cardboard, composite_carton, green_glass,
hazardous_waste_(battery), metal, non_waste, organic_waste, paper,
pet, plastic, plastic_aluminium, residual_waste, rigid_plastic_container,
white_glass, white_glass_metal
```

## Model Performance

| Metric | Value |
|---|---|
| Test Accuracy | 93.23% |
| Test Loss | 0.3379 |
| Macro F1 | 0.932 |
| Weighted F1 | 0.932 |

Training used a two-phase procedure: Phase 1 trains the classifier head with the
backbone frozen; Phase 2 fine-tunes the full network. Early stopping was applied
in both phases. Total epochs trained: **[INSERT FINAL EPOCH COUNT]**.

Best-performing classes (100% accuracy): `aluminium`, `hazardous_waste_(battery)`,
`organic_waste`. Weakest class: `plastic` (77.05%).

## Project Structure

```
ba_agentic_framework/
├── app.py                               # Gunicorn entry point for HF Spaces
├── requirements.txt
└── waste_recycling_chatbot_pa2/
    ├── chatbot/
    │   ├── swiss_waste_agent.py         # LangGraph agent, nodes, routing
    │   └── knowledge_base.py            # RECYCLING_GUIDE data, WasteClassifier
    ├── dashboard/
    │   └── dashboard_app.py             # Dash UI and Flask server
    ├── config/
    │   └── config.py                    # Path configuration for local runs
    ├── notebooks/
    │   └── 02_baseline_training.ipynb   # MobileNetV3 training pipeline
    └── scripts/
        └── upload_model.py              # Uploads trained model to HF Hub
```

## Installation and Local Setup

Requirements: Python 3.10+

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root containing your OpenAI API key:

```
OPENAI_API_KEY=your_key_here
```

The trained classifier is downloaded automatically from Hugging Face Hub on first
run. No manual model download is required.

## Running the App

Start the Dash development server:

```bash
python waste_recycling_chatbot_pa2/dashboard/dashboard_app.py
```

The app opens at `http://127.0.0.1:8050/` in the default browser.

The app is also deployed on Hugging Face Spaces and accessible there without any
local setup.

## Credits

Author: Lejla Beganovic  
Supervisor: Martin Schüle  
ZHAW Life Sciences und Facility Management  
Institut für Computational Life Sciences, Wädenswil
