"""
API Model Library — per-token inference pricing
Pricing as of March 27, 2026

Sources: openai.com/api/pricing, docs.anthropic.com, ai.google.dev/gemini-api/docs/pricing, openrouter.ai
"""

# Prices are in $ per 1M tokens
MODEL_LIBRARY = {
    "GPT-5.4 Pro":           {"provider": "OpenAI",    "input": 30,    "output": 180,   "notes": "Top-tier reasoning model, Mar 2026"},
    "GPT-5.4":               {"provider": "OpenAI",    "input": 2.5,   "output": 15,    "notes": "Flagship, Feb 2026"},
    "GPT-5.4 Mini":          {"provider": "OpenAI",    "input": 0.75,  "output": 4.5,   "notes": "Mid-tier 5.4 variant, Mar 2026"},
    "GPT-5.4 Nano":          {"provider": "OpenAI",    "input": 0.20,  "output": 1.25,  "notes": "Efficient 5.4 variant, Mar 2026"},
    "GPT-5.3":               {"provider": "OpenAI",    "input": 1.75,  "output": 14,    "notes": "Chat-optimized, Mar 2026"},
    "GPT-5.2 Pro":           {"provider": "OpenAI",    "input": 10.5,  "output": 84,    "notes": "Reasoning variant, 400K context (price halved Mar 2026)"},
    "GPT-5.2":               {"provider": "OpenAI",    "input": 0.875, "output": 7,     "notes": "Dec 2025 flagship (price halved Mar 2026)"},
    "GPT-5.1":               {"provider": "OpenAI",    "input": 0.625, "output": 5,     "notes": "Coding-optimized (price halved Mar 2026)"},
    "GPT-5":                 {"provider": "OpenAI",    "input": 0.625, "output": 5,     "notes": "Aug 2025 flagship, 400K context (price halved Mar 2026)"},
    "GPT-5 Mini":            {"provider": "OpenAI",    "input": 0.25,  "output": 2,     "notes": "Efficient mid-tier, great value"},
    "GPT-5 Nano":            {"provider": "OpenAI",    "input": 0.05,  "output": 0.4,   "notes": "Cheapest OpenAI option"},
    "GPT-4.1":               {"provider": "OpenAI",    "input": 2,     "output": 8,     "notes": "Strong all-rounder, 1M context"},
    "GPT-4.1 Mini":          {"provider": "OpenAI",    "input": 0.2,   "output": 0.8,   "notes": "Efficient mid-tier, 1M context (price halved Mar 2026)"},
    "GPT-4.1 Nano":          {"provider": "OpenAI",    "input": 0.05,  "output": 0.2,   "notes": "Fastest & cheapest GPT-4.1 (price halved Mar 2026)"},
    "o3":                    {"provider": "OpenAI",    "input": 2,     "output": 8,     "notes": "Reasoning model, price dropped Mar 2026"},
    "o3-mini":               {"provider": "OpenAI",    "input": 1.1,   "output": 4.4,   "notes": "Affordable reasoning model"},
    "o4-mini":               {"provider": "OpenAI",    "input": 1.1,   "output": 4.4,   "notes": "Affordable reasoning model"},
    "o1":                    {"provider": "OpenAI",    "input": 15,    "output": 60,    "notes": "Legacy reasoning, high-cost"},
    "Claude Opus 4.6":       {"provider": "Anthropic", "input": 5,     "output": 25,    "notes": "Most capable Anthropic model"},
    "Claude Sonnet 4.6":     {"provider": "Anthropic", "input": 3,     "output": 15,    "notes": "Opus-level performance at Sonnet pricing, 1M context"},
    "Claude Haiku 4.5":      {"provider": "Anthropic", "input": 1,     "output": 5,     "notes": "Fast & efficient, great for routing"},
    "Claude Opus 4.5":       {"provider": "Anthropic", "input": 5,     "output": 25,    "notes": "Previous flagship, same pricing as 4.6"},
    "Claude Sonnet 4.5":     {"provider": "Anthropic", "input": 3,     "output": 15,    "notes": "Previous Sonnet, same pricing as 4.6"},
    "Claude Sonnet 4":       {"provider": "Anthropic", "input": 3,     "output": 15,    "notes": "Previous generation Sonnet"},
    "Claude Haiku 3":        {"provider": "Anthropic", "input": 0.25,  "output": 1.25,  "notes": "Retiring Apr 2026"},
    "Gemini 3.1 Pro":        {"provider": "Google",    "input": 2,     "output": 12,    "notes": "Latest Google flagship, Mar 2026"},
    "Gemini 3 Flash":        {"provider": "Google",    "input": 0.5,   "output": 3,     "notes": "Pro-grade reasoning at Flash speed"},
    "Gemini 3.1 Flash Lite": {"provider": "Google",    "input": 0.25,  "output": 1.5,   "notes": "Cost-efficient 3.1 variant, Mar 2026"},
    "Gemini 2.5 Pro":        {"provider": "Google",    "input": 1.25,  "output": 10,    "notes": "Production-ready, 1M context"},
    "Gemini 2.5 Flash":      {"provider": "Google",    "input": 0.3,   "output": 2.5,   "notes": "Capable budget option (repriced Mar 2026)"},
    "Gemini 2.5 Flash-Lite": {"provider": "Google",    "input": 0.1,   "output": 0.4,   "notes": "Cost-efficient, now GA"},
    "Gemini 2.0 Flash-Lite": {"provider": "Google",    "input": 0.075, "output": 0.3,   "notes": "Cheapest Google model, retiring Jun 2026"},
    "Grok 4.20":             {"provider": "xAI",       "input": 2,     "output": 6,     "notes": "New flagship, 2M context, Mar 2026"},
    "Grok 4.1 Fast":         {"provider": "xAI",       "input": 0.2,   "output": 0.5,   "notes": "2M context, very competitive pricing"},
    "Mistral Large 3":       {"provider": "Mistral",   "input": 0.5,   "output": 1.5,   "notes": "675B params, via Mistral API"},
    "Mistral Small 4":       {"provider": "Mistral",   "input": 0.15,  "output": 0.6,   "notes": "Hybrid reasoning, multimodal, 262K context. Via OpenRouter."},
    "DeepSeek V3.2":         {"provider": "DeepSeek",  "input": 0.28,  "output": 0.42,  "notes": "Cost-effective API, strong coding/math"},
    "Qwen3 Max":             {"provider": "Alibaba",   "input": 0.78,  "output": 3.9,   "notes": "Qwen flagship, 262K context. Via OpenRouter."},
    "Qwen3.5 Plus":          {"provider": "Alibaba",   "input": 0.26,  "output": 1.56,  "notes": "Qwen mid-tier. Via OpenRouter."},
    "Qwen3.5 397B A17B":     {"provider": "Alibaba",   "input": 0.39,  "output": 2.34,  "notes": "Open-weights 397B MoE (17B active), vision-language. Via OpenRouter."},
    "Qwen3 235B A22B":       {"provider": "Alibaba",   "input": 0.455, "output": 1.82,  "notes": "Open-weights 235B MoE (22B active), Instruct. Via OpenRouter."},
    "Kimi K2.5":             {"provider": "Moonshot",   "input": 0.45,  "output": 2.2,   "notes": "Strong coding & math, 262K context. Via OpenRouter."},
    "MiniMax M2.7":          {"provider": "MiniMax",    "input": 0.3,   "output": 1.2,   "notes": "Latest MiniMax flagship. Via OpenRouter."},
    "MiniMax M2.5":          {"provider": "MiniMax",    "input": 0.2,   "output": 1.17,  "notes": "Previous MiniMax flagship. Via OpenRouter."},
    "MiniMax M2-Her":        {"provider": "MiniMax",    "input": 0.3,   "output": 1.2,   "notes": "65K context. Via OpenRouter."},
    "Llama 4 Maverick":      {"provider": "Meta",       "input": 0.15,  "output": 0.6,   "notes": "Open-weights 400B MoE (17B active). Via OpenRouter."},
    "Llama 4 Scout":         {"provider": "Meta",       "input": 0.08,  "output": 0.3,   "notes": "Open-weights, efficient Llama 4 variant. Via OpenRouter."},
    "Arcee Trinity Large":   {"provider": "Arcee AI",   "input": 0,     "output": 0,     "notes": "Open-weights 400B MoE (13B active), 512K context. Preview, free on OpenRouter."},
    "Arcee Trinity Mini":    {"provider": "Arcee AI",   "input": 0.045, "output": 0.15,  "notes": "Open-weights 26B MoE (3B active), 128K context. Via OpenRouter."},
    "Arcee Trinity Nano":    {"provider": "Arcee AI",   "input": None,  "output": None,  "notes": "Open-weights 6B MoE (1B active), 128K context. Self-hosted only."},
}

API_MODELS = [name for name, m in MODEL_LIBRARY.items() if m["input"] is not None]
