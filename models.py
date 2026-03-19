"""
API Model Library — per-token inference pricing
Pricing as of March 2026

Sources: openai.com/api/pricing, docs.anthropic.com, ai.google.dev/gemini-api/docs/pricing, openrouter.ai
"""

# Prices are in $ per 1M tokens
MODEL_LIBRARY = {
    "GPT-5.4 Pro":           {"provider": "OpenAI",    "input": 30,    "output": 180,   "notes": "Top-tier reasoning model, Mar 2026"},
    "GPT-5.4":               {"provider": "OpenAI",    "input": 2.5,   "output": 15,    "notes": "Latest flagship, Feb 2026"},
    "GPT-5.2 Pro":           {"provider": "OpenAI",    "input": 21,    "output": 168,   "notes": "High-end reasoning variant, 400K context"},
    "GPT-5.2":               {"provider": "OpenAI",    "input": 1.75,  "output": 14,    "notes": "Previous flagship, Dec 2025"},
    "GPT-5.1":               {"provider": "OpenAI",    "input": 1.25,  "output": 10,    "notes": "Coding-optimized variant"},
    "GPT-5":                 {"provider": "OpenAI",    "input": 1.25,  "output": 10,    "notes": "Released Aug 2025, 400K context"},
    "GPT-5 Mini":            {"provider": "OpenAI",    "input": 0.25,  "output": 2,     "notes": "Efficient mid-tier, great value"},
    "GPT-5 Nano":            {"provider": "OpenAI",    "input": 0.05,  "output": 0.4,   "notes": "Cheapest OpenAI option"},
    "GPT-4.1":               {"provider": "OpenAI",    "input": 2,     "output": 8,     "notes": "Strong all-rounder, 1M context"},
    "GPT-4.1 Mini":          {"provider": "OpenAI",    "input": 0.4,   "output": 1.6,   "notes": "Efficient mid-tier, 1M context"},
    "GPT-4.1 Nano":          {"provider": "OpenAI",    "input": 0.1,   "output": 0.4,   "notes": "Fastest & cheapest GPT-4.1"},
    "o3":                    {"provider": "OpenAI",    "input": 2,     "output": 8,     "notes": "Reasoning model, price dropped Mar 2026"},
    "o4-mini":               {"provider": "OpenAI",    "input": 1.1,   "output": 4.4,   "notes": "Affordable reasoning model"},
    "o1":                    {"provider": "OpenAI",    "input": 15,    "output": 60,    "notes": "Legacy reasoning, high-cost"},
    "Claude Opus 4.6":       {"provider": "Anthropic", "input": 5,     "output": 25,    "notes": "Most capable Anthropic model"},
    "Claude Sonnet 4.6":     {"provider": "Anthropic", "input": 3,     "output": 15,    "notes": "Opus-level performance at Sonnet pricing, 1M context"},
    "Claude Haiku 4.5":      {"provider": "Anthropic", "input": 1,     "output": 5,     "notes": "Fast & efficient, great for routing"},
    "Claude Opus 4.5":       {"provider": "Anthropic", "input": 5,     "output": 25,    "notes": "Previous flagship, same pricing as 4.6"},
    "Claude Sonnet 4.5":     {"provider": "Anthropic", "input": 3,     "output": 15,    "notes": "Previous Sonnet, same pricing as 4.6"},
    "Claude Sonnet 4":       {"provider": "Anthropic", "input": 3,     "output": 15,    "notes": "Previous generation Sonnet"},
    "Claude Haiku 3.5":      {"provider": "Anthropic", "input": 0.8,   "output": 4,     "notes": "Legacy efficient model"},
    "Claude Haiku 3":        {"provider": "Anthropic", "input": 0.25,  "output": 1.25,  "notes": "Cheapest Anthropic option"},
    "Gemini 3.1 Pro":        {"provider": "Google",    "input": 2,     "output": 12,    "notes": "Latest Google flagship, Mar 2026"},
    "Gemini 3 Pro":          {"provider": "Google",    "input": 2,     "output": 12,    "notes": "Previous flagship"},
    "Gemini 3 Flash":        {"provider": "Google",    "input": 0.5,   "output": 3,     "notes": "Pro-grade reasoning at Flash speed"},
    "Gemini 2.5 Pro":        {"provider": "Google",    "input": 1.25,  "output": 10,    "notes": "Production-ready, 1M context"},
    "Gemini 2.5 Flash":      {"provider": "Google",    "input": 0.15,  "output": 0.6,   "notes": "Best budget option, very capable"},
    "Gemini 2.5 Flash-Lite": {"provider": "Google",    "input": 0.1,   "output": 0.4,   "notes": "Ultra-low-cost Google option"},
    "Gemini 3.1 Flash Lite": {"provider": "Google",    "input": 0.25,  "output": 1.5,   "notes": "Cost-efficient 3.1 variant, Mar 2026"},
    "Gemini 2.0 Flash-Lite": {"provider": "Google",    "input": 0.075, "output": 0.3,   "notes": "Cheapest Google model"},
    "Grok 4":                {"provider": "xAI",       "input": 3,     "output": 15,    "notes": "256K context, Jul 2025"},
    "Grok 4.1 Fast":         {"provider": "xAI",       "input": 0.2,   "output": 0.5,   "notes": "2M context, very competitive pricing"},
    "Mistral Large 3":       {"provider": "Mistral",   "input": 0.5,   "output": 1.5,   "notes": "675B params, available on OpenRouter"},
    "DeepSeek V4":           {"provider": "DeepSeek",  "input": 0.3,   "output": 0.5,   "notes": "Latest DeepSeek, Mar 2026, 1M context"},
    "DeepSeek V3.2":         {"provider": "DeepSeek",  "input": 0.28,  "output": 0.42,  "notes": "Cost-effective API, strong coding/math"},
    "Qwen3 Max":             {"provider": "Alibaba",   "input": 1.2,   "output": 6,     "notes": "Latest Qwen flagship, 262K context. Via OpenRouter."},
    "Qwen3.5 Plus":          {"provider": "Alibaba",   "input": 0.26,  "output": 1.56,  "notes": "Latest Qwen mid-tier. Via OpenRouter."},
    "Qwen3 235B A22B":       {"provider": "Alibaba",   "input": 0.07,  "output": 0.1,   "notes": "Open-weights 235B MoE (22B active). Via OpenRouter."},
    "Kimi K2.5":             {"provider": "Moonshot",   "input": 0.45,  "output": 2.2,   "notes": "Strong coding & math, 262K context. Via OpenRouter."},
    "MiniMax M2.5":          {"provider": "MiniMax",    "input": 0.27,  "output": 0.95,  "notes": "Latest MiniMax flagship. Via OpenRouter."},
    "MiniMax M2-Her":        {"provider": "MiniMax",    "input": 0.3,   "output": 1.2,   "notes": "65K context. Via OpenRouter."},
    "Llama 4 Maverick":      {"provider": "Meta",       "input": 0.15,  "output": 0.6,   "notes": "Open-weights 400B MoE (17B active). Via OpenRouter."},
    "Llama 4 Scout":         {"provider": "Meta",       "input": 0.08,  "output": 0.3,   "notes": "Open-weights, efficient Llama 4 variant. Via OpenRouter."},
    "Arcee Trinity Large":   {"provider": "Arcee AI",   "input": 0,     "output": 0,     "notes": "Open-weights 400B MoE (13B active), 512K context. Preview, free on OpenRouter."},
    "Arcee Trinity Mini":    {"provider": "Arcee AI",   "input": 0.045, "output": 0.15,  "notes": "Open-weights 26B MoE (3B active), 128K context. Via OpenRouter."},
    "Arcee Trinity Nano":    {"provider": "Arcee AI",   "input": None,  "output": None,  "notes": "Open-weights 6B MoE (1B active), 128K context. Self-hosted only."},
}

API_MODELS = [name for name, m in MODEL_LIBRARY.items() if m["input"] is not None]
