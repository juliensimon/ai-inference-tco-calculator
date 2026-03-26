---
title: AI Inference TCO Calculator
emoji: 💰
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "5.12.0"
app_file: app.py
pinned: true
---

# AI Inference TCO Calculator

**By [Julien Simon](https://www.linkedin.com/in/juliensimon/) | AI Operating Partner, [Fortino Capital](https://fortinocapital.com)**

> **[Try it live on Hugging Face Spaces](https://huggingface.co/spaces/juliensimon/tco-simulator)**

How much does it *really* cost to run inference in production? This calculator compares the Total Cost of Ownership across three deployment options — API, self-hosted GPU, and local/edge — so you can make informed build-vs-buy decisions for your inference workloads.

<p align="center">
  <img src="https://img.shields.io/badge/Models-53-blue" alt="53 models">
  <img src="https://img.shields.io/badge/GPU%20Instances-56-green" alt="56 GPU instances">
  <img src="https://img.shields.io/badge/GPU%20Providers-10-orange" alt="10 providers">
  <img src="https://img.shields.io/badge/Pricing-March%2026%2C%202026-red" alt="March 26, 2026 pricing">
</p>

## Inference Deployment Options Compared

| Option | Description | Best For |
|--------|-------------|----------|
| **API Providers** | Pay-per-token inference via Claude, GPT, Gemini, Grok, DeepSeek, and 40+ models | Low volume, fast iteration, no infra team |
| **Self-Hosted GPU** | Run your own inference stack (vLLM, TGI, etc.) on cloud GPUs | High volume, data privacy, cost optimization |
| **Local / Edge** | On-premises inference with consumer hardware (RTX 5090, etc.) | Small models, ultra-low latency, air-gapped |

## Features

- **52 inference API models** with current per-token pricing from OpenAI, Anthropic, Google, xAI, DeepSeek, Meta, Mistral, and more
- **57 GPU instances** for self-hosted inference across 10 cloud providers (AWS, GCP, Azure, CoreWeave, Lambda, RunPod, Crusoe, Together AI, Vast.ai, FluidStack)
- **8 GPU types**: L4, L40S, A100, H100, H200, B200, GB200, MI300X (NVIDIA + AMD)
- **Smart routing** scenario (60/40 cheapest blend across inference providers)
- **Break-even analysis** showing the daily request volume where self-hosted inference beats API
- **Real-time calculations** — all charts and tables update as you adjust parameters
- **Interactive Plotly charts** with cost breakdowns and side-by-side comparisons

## Quick Start

### Use Online

Visit the **[Hugging Face Space](https://huggingface.co/spaces/juliensimon/tco-simulator)** — no installation needed.

### Run Locally

```bash
git clone https://github.com/juliensimon/ai-inference-tco-calculator.git
cd ai-inference-tco-calculator
pip install -r requirements.txt
python app.py
```

### Run Tests

```bash
pip install pytest
pytest test_app.py -v
```

## How to Use

1. **Your Inputs** — Set your inference workload parameters (tokens per request, requests per day, etc.)
2. **API Costs** — Select up to 4 API providers/models; prices auto-populate from the Model Library
3. **Self-Hosted GPU** — Pick a cloud provider and GPU instance from the dropdowns, or enter custom pricing
4. **Local / Edge** — Configure on-premises hardware parameters
5. **Comparison** — View side-by-side annual costs, per-million-token costs, and break-even analysis
6. **Model Library** — Browse all 52 models with current pricing
7. **GPU Library** — Browse all 57 GPU instances with per-hour pricing across providers

## Project Structure

| File | Description |
|------|-------------|
| `app.py` | UI, calculations, charts, event wiring |
| `models.py` | API model library (52 models with per-token pricing) |
| `gpus.py` | GPU instance library (57 instances across 10 providers) |
| `test_app.py` | 81 unit tests covering helpers, calculations, and integration |
| `SOURCES.md` | All pricing data sources for future updates |

## Data Sources

See [SOURCES.md](SOURCES.md) for the full list of pricing sources and aggregator tools.

### API Model Pricing
openai.com, docs.anthropic.com, ai.google.dev, openrouter.ai

### GPU Instance Pricing
aws.amazon.com, cloud.google.com, azure.microsoft.com, coreweave.com, crusoe.ai, fluidstack.io, lambda.ai, runpod.io, together.ai, vast.ai

All pricing as of March 22, 2026. Contributions welcome to keep pricing current.

## Contributing

Pricing changes fast. To update:
- **API models** — edit `models.py` (`MODEL_LIBRARY` dict)
- **GPU instances** — edit `gpus.py` (`GPU_LIBRARY` dict)

Please include your data source in the PR. Run `pytest test_app.py` before submitting.

## License

MIT
