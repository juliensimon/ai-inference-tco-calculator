# Data Sources

Last updated: July 9, 2026

## API Model Pricing (`models.py`)

| Provider | Source |
|----------|--------|
| OpenAI | [developers.openai.com/api/docs/pricing](https://developers.openai.com/api/docs/pricing) (legacy models via OpenRouter) |
| Anthropic | [platform.claude.com/docs](https://platform.claude.com/docs/en/docs/about-claude/models) |
| Google | [ai.google.dev/gemini-api/docs/pricing](https://ai.google.dev/gemini-api/docs/pricing) |
| xAI | [docs.x.ai/docs/models](https://docs.x.ai/docs/models) |
| DeepSeek | [api-docs.deepseek.com](https://api-docs.deepseek.com/quick_start/pricing) |
| Mistral, Qwen, Kimi, MiniMax, Meta, Arcee | [openrouter.ai](https://openrouter.ai) (prices via /api/v1/models) |

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
- Vast.ai prices are marketplace medians and fluctuate
- All prices are on-demand unless noted otherwise
- **Regions used:** AWS us-east-1, GCP us-central1, Azure East US (H200/MI300X: East US 2), CoreWeave US-East, Crusoe us-north1, Lambda US, RunPod US, Together US, Vast.ai US (marketplace)
- FluidStack removed July 2026: public on-demand pricing page discontinued (contract-only sales)
