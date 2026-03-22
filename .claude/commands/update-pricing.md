Refresh API model and GPU instance pricing for the AI Inference TCO Calculator.

## Process

### 1. Read Current State

Read these files to understand current data:
- `models.py` — MODEL_LIBRARY dict
- `gpus.py` — GPU_LIBRARY dict
- `SOURCES.md` — all pricing data sources

### 2. Research Pricing

Dispatch **two parallel research agents**:

**Agent 1 — API Models** (`models.py`):
- Check each provider's pricing page (see SOURCES.md)
- Look for new models released since the last update
- Look for deprecated/removed models
- Note any price changes

**Agent 2 — GPU Instances** (`gpus.py`):
- Check each provider's pricing page (see SOURCES.md)
- Use aggregators (instances.vantage.sh, computeprices.com, getdeploying.com) to cross-reference
- For multi-GPU nodes: per-GPU price = total node price / GPU count
- Look for new instance types or GPU generations
- Vast.ai uses marketplace median pricing

### 3. Verify Agent Findings Against Primary Sources

**CRITICAL: Research agents hallucinate pricing data.** Before applying ANY changes, spot-check the agent's findings by fetching primary provider pages directly with WebFetch:

- For **API models**: Fetch the provider's own pricing page (e.g. `openai.com/api/pricing`, `lambda.ai/pricing`). If Cloudflare-blocked, use `pricepertoken.com/pricing-page/provider/[name]` as a cross-reference.
- For **GPU instances**: Fetch at least the top 3 providers with reported changes (e.g. `lambda.ai/pricing`, `crusoe.ai/cloud/pricing`, `runpod.io/pricing`).
- **Reject any change** where the agent's price differs from the primary source. Agents commonly hallucinate price cuts and non-existent instances.
- For Vast.ai marketplace prices, accept reasonable fluctuation but verify the direction of change.

### 4. Apply Updates

- Update `models.py` — add/remove/update entries
- Update `gpus.py` — add/remove/update entries
- Update docstring dates in both files
- Keep entries sorted: by provider (alpha), then by GPU type within provider
- Key format for GPUs: `"Provider - GPU - instance_type"` or `"Provider - GPU"`

### 5. Run Tests

```bash
pytest test_app.py -v
python -c "from models import MODEL_LIBRARY, API_MODELS; from gpus import GPU_LIBRARY, GPU_PROVIDERS; print(f'{len(MODEL_LIBRARY)} models, {len(API_MODELS)} API, {len(GPU_LIBRARY)} GPUs, {len(GPU_PROVIDERS)} providers')"
```

### 6. Update Metadata

- Update `SOURCES.md` "Last updated" date
- Update README.md badge counts if model/GPU counts changed
- Update "Pricing as of" dates in file docstrings

### 7. Ship

Commit and push to both remotes:
```bash
git add models.py gpus.py SOURCES.md README.md
git commit -m "Update pricing to [MONTH YEAR] ([N] models, [N] GPU instances)"
git push origin main && git push hf main
```
