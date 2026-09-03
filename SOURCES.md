# Data Sources

Last updated: September 3, 2026

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
| Mistral | [mistral.ai/pricing/api](https://mistral.ai/pricing/api/) (first-party) |
| Zhipu (GLM) | [docs.z.ai/guides/overview/pricing](https://docs.z.ai/guides/overview/pricing) (first-party; GLM 5.2 also hosted on Mistral La Plateforme at the same price) |
| MiniMax, Meta, open-weights Qwen | [openrouter.ai](https://openrouter.ai) (prices via /api/v1/models) |

### Chinese-provider notes

- **DeepSeek peak/off-peak pricing took effect August 2026.** Peak hours are 01:00–04:00 and 06:00–10:00 UTC, **Monday through Friday**, at 2x the off-peak rate; weekends are entirely off-peak. The library stores the **off-peak** (17h/day) cache-miss rate, with peak rates in each model's `notes`.
- **DeepSeek, Alibaba, Moonshot and Zhipu are priced from their own first-party APIs**, not OpenRouter. At the Aug 19, 2026 refresh OpenRouter showed Kimi K2.7 Code at $0.71/$3.50 while Moonshot's own page still said $0.95/$4.00 — another confirmed drift case. OpenRouter resells these at a markup and its rates drift with routing — it disagreed with first-party pricing on Qwen3.7 Max and Kimi K2.7 Code at the July 2026 refresh.
- **Alibaba Model Studio prices are context-tiered.** The library stores the base tier; the tier boundary and higher rate are recorded in each model's `notes`.
- **Promotional pricing: the library always stores the rate you would actually be billed today**, with any post-promo list price recorded in the model's `notes`. This holds regardless of whether the promo has a published end date. Alibaba's open-ended discount (Qwen3.7 Plus 20% off — the Qwen3.7 Max 50% promo ended by Sep 2, 2026 and that entry now stores list $2.50/$7.50), Z.ai's dated promo (GLM 5.3 Flash 50% off through Sep 9, 2026), Google's dated promo (Gemini 3.7/3.6 Flash at $0.75/$3.75 through Dec 31, 2026, rising to $1.50/$7.50 on Jan 1, 2027) and OpenAI's floor-only promo (GPT-5.6 Sol at $4/$20, "at least through November 21, 2026", with no list price published at all) are all stored at the **effective** rate. Rationale: this is a cost calculator, so a number nobody is charged today is the wrong default — it read ~2x high for Gemini before Aug 25, 2026. Claude Sonnet 5's $2/$10 introductory rate became permanent when the Sep 2026 rise to $3/$15 was cancelled, so its effective and list prices now coincide.
- `qwen3.7-flash` was China (Beijing) region only until Aug 2026; it is now on the International/Singapore list at the same base tier ($0.03/$0.13 ≤32K). Model Studio also now lists open-weights Qwen (3.8 27B, 3.8 2.4T, 3.5 397B, 3 235B) first-party; the library keeps pricing those via OpenRouter so figures stay comparable across refreshes, noting the first-party rate where it differs (Qwen3.5 397B: Model Studio $0.60/$3.60 vs OpenRouter $0.55/$3.50).
- **Closed-weights models leave the library when their sole provider retires them** (Mistral Medium 3, retired Aug 31, 2026), even if OpenRouter still shows a price. Open-weights models retired by their original provider stay, priced via third-party hosts on OpenRouter (Kimi K2.5, retired by Moonshot Aug 31, 2026).
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
- **CoreWeave L40** (8-GPU node, $10.00/hr, 48GB) is tracked alongside L40S. Verified by the same node-division method that reproduces L40S exactly ($18.00/8 = $2.25).
- **AWS, GCP and Azure are tracked at full instance-size granularity** (AWS: every size of p4d/p4de/p5/p5en/p6/g6/gr6/g6e/g7/g7e in us-east-1; GCP: a2/a3/g2/g4 in us-central1; Azure: NC/ND GPU families in East US / East US 2). Fractional-GPU sizes are excluded everywhere (AWS g6f/gr6f, GCP g4-standard-6/12/24, Azure NC36ds/NC72ds and lds twins — 1/8 to 1/2 of a GPU does not fit the per-GPU cost schema). GPU counts per size verified against provider size documentation.
- **Azure's retail API lists RTX PRO 6000 sizes that are absent from Microsoft's own size docs** (NC128/132/256/264/320/324 ds and lds) — GPU counts unverifiable, so they are excluded. The pricing ladder is also non-linear (NC288ds $11.00 &lt; NC256ds $12.76): do not infer GPU counts from price.
- Vast.ai prices are marketplace medians and fluctuate. Method: `GET console.vast.ai/api/v0/bundles/` with `q={"verified":{"eq":true},"rentable":{"eq":true},"gpu_name":{"eq":"<name>"}}`, then median of `dph_total / num_gpus`. Query **per GPU type** — the unfiltered endpoint caps at 64 offers regardless of `limit`. Sample size `n` is recorded in each entry's notes; restrict to verified hosts for reproducibility. Repeated queries can return HTTP 429 — back off and retry rather than accepting a partial sample.
- **Vast.ai `A100 SXM4` mixes 40GB and 80GB cards.** Filter on `gpu_ram` or the median is a blend of two products. At the Aug 2026 refresh the blend was $0.84 while true 40GB was $0.80 and true 80GB was $1.13 — the library's old $0.80 "80GB" figure was in fact the blended/40GB number. Other GPU types in the library are VRAM-homogeneous.
- **Vast.ai split `RTX PRO 6000` into `RTX PRO 6000 S` (Server Edition) and `RTX PRO 6000 WS` (Workstation)**; the bare name now returns zero offers. The library tracks the **S** variant, since AWS/Azure/GCP/CoreWeave/RunPod all list Server Edition.
- Medians with an even sample size can oscillate between refreshes as offers flip rentable state (Vast L40S moved between $0.74 and $0.80 within minutes). Sample more than once before recording a change. At the Aug 25, 2026 refresh a research agent reported RTX PRO 6000 S at $1.40; five independent samples all returned $1.5196 (n=20), so the $1.40 was rejected. At the Sep 3, 2026 refresh B200 fell from $7.75 to $6.01 as n dropped from 19 to 6–7 and the offer set turned bimodal (~$6.00 and ~$9.99); five samples over ~20 minutes agreed, so it was accepted, but expect it to move again.
- **Vast.ai medians are global, not US.** The documented query filters on `verified`/`rentable`/`gpu_name` only — there is no geo predicate — so every stored Vast figure is a worldwide median. This was previously mislabelled "Vast.ai US". The label is not merely imprecise: **L4 and L40S have zero US offers**, so a US-only median does not exist for them, and the stored $0.33/$0.80 match the global medians exactly. Geography moves the number materially — at this refresh H100 SXM was $3.07 global against $4.77 US-only. Keep using the global method so figures stay comparable across refreshes.

### Sources that gave wrong numbers (do not use)

- **ec2.shop** — stale for AWS P-series; returns pre-June-2025 list prices (p5.48xlarge at 98.32 vs the true 55.04) and omits p6 entirely. Its G-series figures are correct, which makes the error easy to miss. Use the AWS pricing feed instead: `curl --compressed 'https://b0.p.awsstatic.com/pricing/2.0/meteredUnitMaps/ec2/USD/current/ec2-ondemand-without-sec-sel/US%20East%20(N.%20Virginia)/Linux/index.json'` (gzipped — `--compressed` is required or JSON parsing fails).
- **instances.vantage.sh GCP JSON** — excludes accelerator cost for GPU machine types, reporting a3-highgpu-8g at 9.46/hr against a true 88.49. Fine for AWS/Azure; do not use for GCP.
- **mistral.ai/pricing/api** is authoritative for Mistral *prices* but not for *availability* — it still lists Mixtral 8x7B/8x22B (retired March 2025) and Magistral/Devstral/NeMo (retired July 31, 2026). Check [docs.mistral.ai/getting-started/models](https://docs.mistral.ai/getting-started/models/), which carries the deprecation/retirement table, before adding a Mistral model. **But that table is per-snapshot, not per-family** — it lists `codestral-2405` (retired 6/16/2025) and `voxtral-small-25-07` (retired 5/31/2026) while Codestral and Voxtral Small are both live on the pricing page today at unchanged rates. Only retire a library entry when the *family* leaves the pricing page, not when a dated snapshot ages out.
- **GCP `a4-highgpu-8g` (B200) has no on-demand price** — the On-Demand column reads "N/A". The $64.44 shown alongside it is DWS Flex-start; dividing it by 8 yields a plausible-looking $8.06 that is not an on-demand rate. Same trap applies to any row whose On-Demand cell is N/A.
- **Azure's retail API returns ~150 meters per GPU SKU family**, mixing Windows/Linux, Spot, and superseded preview SKU names. Filter to Linux (`productName` without "Windows") and non-Spot `skuName`, and confirm accelerator counts against the size-series doc — `ds` and `lds` variants differ in storage, not GPU count, and preview names can carry stale prices alongside the GA ones. The `lds` (local-disk) sizes are also priced lower (NC144lds RTX PRO 6000 at $5.50 vs NC144ds at $6.38); the library records the `ds` size.
- All prices are on-demand unless noted otherwise
- **Regions used:** AWS us-east-1, GCP us-central1, Azure East US (H200/MI300X: East US 2), CoreWeave US-East, Crusoe us-north1, Lambda US, RunPod US, Together US, Vast.ai **global** (marketplace — see note below)
- FluidStack removed July 2026: public on-demand pricing page discontinued (contract-only sales)
