# Data Sources

Last updated: July 31, 2026

## API Model Pricing (`models.py`)

| Provider | Source |
|----------|--------|
| OpenAI | [developers.openai.com/api/docs/pricing](https://developers.openai.com/api/docs/pricing) (legacy models via OpenRouter) |
| Anthropic | [platform.claude.com/docs](https://platform.claude.com/docs/en/docs/about-claude/models) |
| Google | [ai.google.dev/gemini-api/docs/pricing](https://ai.google.dev/gemini-api/docs/pricing) |
| xAI | [docs.x.ai/docs/models](https://docs.x.ai/docs/models) |
| DeepSeek | [api-docs.deepseek.com](https://api-docs.deepseek.com/quick_start/pricing) (first-party, cache-miss input) |
| Alibaba (Qwen) | [alibabacloud.com/help/en/model-studio/model-pricing](https://www.alibabacloud.com/help/en/model-studio/model-pricing) (Model Studio, International/Singapore, USD) |
| Moonshot (Kimi) | [platform.kimi.ai/docs/pricing](https://platform.kimi.ai/docs/pricing) (first-party, cache-miss input) |
| Mistral, MiniMax, Meta, Arcee, open-weights Qwen | [openrouter.ai](https://openrouter.ai) (prices via /api/v1/models) |

### Chinese-provider notes

- **DeepSeek, Alibaba and Moonshot are priced from their own first-party APIs**, not OpenRouter. OpenRouter resells these at a markup and its rates drift with routing — it disagreed with first-party pricing on Qwen3.7 Max and Kimi K2.7 Code at the July 2026 refresh.
- **Alibaba Model Studio prices are context-tiered.** The library stores the base tier; the tier boundary and higher rate are recorded in each model's `notes`.
- Alibaba runs open-ended promotional discounts (Qwen3.7 Max at 50% off list, Qwen3.7 Plus at 20% off). The library stores the **effective** price because these promos carry no published end date. Contrast with Anthropic's Claude Sonnet 5 introductory rate, which has a hard end date (Aug 31, 2026) and is therefore stored at **list** price with the promo noted.
- Some Qwen models are China (Beijing) region only and absent from the International/Singapore price list — `qwen3.7-flash` is priced via OpenRouter for this reason.
- Models with no published per-token price (subscription- or self-hosted-only) are stored with `input`/`output` of `None` and excluded from `API_MODELS`.

## GPU Instance Pricing (`gpus.py`)

### Provider pricing pages

| Provider | Source |
|----------|--------|
| AWS | [aws.amazon.com/ec2/pricing/on-demand](https://aws.amazon.com/ec2/pricing/on-demand/) |
| GCP | [cloud.google.com/products/compute/pricing/accelerator-optimized](https://cloud.google.com/products/compute/pricing/accelerator-optimized) |
| Azure | [prices.azure.com retail API](https://prices.azure.com/api/retail/prices) |
| Lambda | [lambda.ai/pricing](https://lambda.ai/pricing) |
| CoreWeave | [coreweave.com/pricing](https://www.coreweave.com/pricing) |
| RunPod | [runpod.io/gpu-pricing](https://www.runpod.io/gpu-pricing) |
| Crusoe | [crusoe.ai/cloud/pricing](https://www.crusoe.ai/cloud/pricing) |
| Together AI | [together.ai/pricing](https://www.together.ai/pricing) |
| Vast.ai | [console.vast.ai/api/v0/search/asks](https://vast.ai) (marketplace medians) |

### Aggregators and comparison tools

| Tool | URL | Notes |
|------|-----|-------|
| Vantage | [instances.vantage.sh](https://instances.vantage.sh) | AWS/Azure/GCP per-GPU breakdowns |
| CloudPrice | [cloudprice.net](https://cloudprice.net) | GCP instance pricing |
| ComputePrices | [computeprices.com](https://computeprices.com) | Multi-provider (good for Vast.ai, FluidStack) |
| Spare Cores | [sparecores.com](https://sparecores.com) | Detailed specs and pricing |
| Economize | [economize.cloud](https://www.economize.cloud) | GCP/AWS instance pricing |
| GetDeploying | [getdeploying.com/gpus](https://getdeploying.com/gpus) | Per-GPU pricing comparison |
| ThunderCompute | [thundercompute.com/blog](https://www.thundercompute.com/blog) | H100/H200/MI300X pricing articles |

## Notes

- For multi-GPU nodes, per-GPU price = total node price / number of GPUs
- Vast.ai prices are marketplace medians and fluctuate. Method: `GET console.vast.ai/api/v0/bundles/` with `q={"verified":{"eq":true},"rentable":{"eq":true},"gpu_name":{"eq":"<name>"}}`, then median of `dph_total / num_gpus`. Query **per GPU type** — the unfiltered endpoint caps at 64 offers regardless of `limit`. Sample size `n` is recorded in each entry's notes; restrict to verified hosts for reproducibility.

### Sources that gave wrong numbers (do not use)

- **ec2.shop** — stale for AWS P-series; returns pre-June-2025 list prices (p5.48xlarge at 98.32 vs the true 55.04) and omits p6 entirely. Its G-series figures are correct, which makes the error easy to miss. Use the AWS pricing feed instead: `curl --compressed 'https://b0.p.awsstatic.com/pricing/2.0/meteredUnitMaps/ec2/USD/current/ec2-ondemand-without-sec-sel/US%20East%20(N.%20Virginia)/Linux/index.json'` (gzipped — `--compressed` is required or JSON parsing fails).
- **instances.vantage.sh GCP JSON** — excludes accelerator cost for GPU machine types, reporting a3-highgpu-8g at 9.46/hr against a true 88.49. Fine for AWS/Azure; do not use for GCP.
- All prices are on-demand unless noted otherwise
- **Regions used:** AWS us-east-1, GCP us-central1, Azure East US (H200/MI300X: East US 2), CoreWeave US-East, Crusoe us-north1, Lambda US, RunPod US, Together US, Vast.ai US (marketplace)
- FluidStack removed July 2026: public on-demand pricing page discontinued (contract-only sales)
