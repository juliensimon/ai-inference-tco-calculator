# Data Sources

Last updated: March 2026

## API Model Pricing (`models.py`)

| Provider | Source |
|----------|--------|
| OpenAI | [openai.com/api/pricing](https://openai.com/api/pricing) |
| Anthropic | [docs.anthropic.com](https://docs.anthropic.com/en/docs/about-claude/models) |
| Google | [ai.google.dev/gemini-api/docs/pricing](https://ai.google.dev/gemini-api/docs/pricing) |
| xAI | [docs.x.ai/docs/models](https://docs.x.ai/docs/models) |
| DeepSeek | [platform.deepseek.com/api-docs/pricing](https://platform.deepseek.com/api-docs/pricing) |
| Mistral, Qwen, Kimi, MiniMax, Meta, Arcee | [openrouter.ai](https://openrouter.ai) |

## GPU Instance Pricing (`gpus.py`)

### Provider pricing pages

| Provider | Source |
|----------|--------|
| AWS | [aws.amazon.com/ec2/pricing/on-demand](https://aws.amazon.com/ec2/pricing/on-demand/) |
| GCP | [cloud.google.com/compute/gpus-pricing](https://cloud.google.com/compute/gpus-pricing) |
| Azure | [azure.microsoft.com/en-us/pricing/details/virtual-machines](https://azure.microsoft.com/en-us/pricing/details/virtual-machines/) |
| Lambda | [lambda.ai/pricing](https://lambda.ai/pricing) |
| CoreWeave | [coreweave.com/pricing](https://www.coreweave.com/pricing) |
| RunPod | [runpod.io/gpu-pricing](https://www.runpod.io/gpu-pricing) |
| Crusoe | [crusoe.ai/cloud/pricing](https://www.crusoe.ai/cloud/pricing) |
| Together AI | [together.ai/pricing](https://www.together.ai/pricing) |
| Vast.ai | [vast.ai](https://vast.ai) (marketplace, variable pricing) |
| FluidStack | [fluidstack.io/pricing](https://www.fluidstack.io/pricing) |

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
