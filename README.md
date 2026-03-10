---
title: AI Infrastructure TCO Calculator
emoji: 💰
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "5.12.0"
app_file: app.py
pinned: true
---

# AI Infrastructure TCO Calculator

**By Julien Simon | AI Operating Partner, Fortino Capital**

Interactive calculator comparing Total Cost of Ownership for AI inference across four deployment options:

- **API providers** (Claude, GPT, Gemini, custom) with smart routing
- **Self-hosted GPU** (cloud H100/H200/B200)
- **Local / Edge** (on-premises consumer GPU/NPU)

## Features

- 39 built-in models with March 2026 pricing
- Real-time cost calculations as you adjust parameters
- Side-by-side comparison with break-even analysis
- Interactive Plotly charts
- Smart routing scenario (60/40 cheapest blend)

## Usage

1. Go to the **Your Inputs** tab and fill in your parameters
2. Select models from dropdowns — prices auto-populate from the Model Library
3. Explore the **API Costs**, **Self-Hosted GPU**, **Local / Edge**, and **Comparison** tabs
4. Check the **Model Library** tab for all available models and pricing

## Run Locally

```bash
pip install -r requirements.txt
python app.py
```
