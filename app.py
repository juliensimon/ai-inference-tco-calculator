"""
AI Infrastructure TCO Calculator
By Julien Simon | AI Operating Partner, Fortino Capital
Pricing as of March 2026
"""

import gradio as gr
import plotly.graph_objects as go
import pandas as pd

# ═════════════════════════════════════════════════════════════════════════════
# MODEL LIBRARY
# ═════════════════════════════════════════════════════════════════════════════

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

# ═════════════════════════════════════════════════════════════════════════════
# GPU INSTANCE LIBRARY
# ═════════════════════════════════════════════════════════════════════════════

GPU_LIBRARY = {
    # ── AWS ──────────────────────────────────────────────────────────────────
    "AWS - A100 40GB - p4d.24xlarge":       {"provider": "AWS",       "gpu": "A100",  "cost_hr": 2.75,  "vram_gb": 40,  "notes": "8-GPU node, ~$21.96/hr total"},
    "AWS - B200 - p6-b200.48xlarge":        {"provider": "AWS",       "gpu": "B200",  "cost_hr": 14.24, "vram_gb": 192, "notes": "8-GPU node, ~$113.93/hr total"},
    "AWS - H100 SXM - p5.48xlarge":         {"provider": "AWS",       "gpu": "H100",  "cost_hr": 6.88,  "vram_gb": 80,  "notes": "8-GPU node, ~$55.04/hr total"},
    "AWS - H200 - p5en.48xlarge":           {"provider": "AWS",       "gpu": "H200",  "cost_hr": 7.91,  "vram_gb": 141, "notes": "8-GPU node, ~$63.30/hr total"},
    "AWS - L4 - g6.xlarge":                 {"provider": "AWS",       "gpu": "L4",    "cost_hr": 0.81,  "vram_gb": 24,  "notes": "1-GPU instance"},
    "AWS - L4 - g6.12xlarge":               {"provider": "AWS",       "gpu": "L4",    "cost_hr": 1.15,  "vram_gb": 24,  "notes": "4-GPU node, ~$4.60/hr total"},
    "AWS - L40S - g6e.xlarge":              {"provider": "AWS",       "gpu": "L40S",  "cost_hr": 1.86,  "vram_gb": 48,  "notes": "1-GPU instance"},
    "AWS - L40S - g6e.12xlarge":            {"provider": "AWS",       "gpu": "L40S",  "cost_hr": 2.62,  "vram_gb": 48,  "notes": "4-GPU node, ~$10.49/hr total"},
    # ── Azure ────────────────────────────────────────────────────────────────
    "Azure - A100 80GB - NC A100 v4":       {"provider": "Azure",     "gpu": "A100",  "cost_hr": 3.67,  "vram_gb": 80,  "notes": "1-GPU instance"},
    "Azure - H100 - ND H100 v5":            {"provider": "Azure",     "gpu": "H100",  "cost_hr": 12.29, "vram_gb": 80,  "notes": "8-GPU node, ~$98.32/hr total"},
    "Azure - H200 - ND H200 v5":            {"provider": "Azure",     "gpu": "H200",  "cost_hr": 13.78, "vram_gb": 141, "notes": "8-GPU node, ~$110.24/hr total"},
    "Azure - MI300X - ND MI300X v5":        {"provider": "Azure",     "gpu": "MI300X", "cost_hr": 7.86, "vram_gb": 192, "notes": "8-GPU node, ~$62.85/hr total"},
    # ── CoreWeave ────────────────────────────────────────────────────────────
    "CoreWeave - A100 80GB":                {"provider": "CoreWeave", "gpu": "A100",  "cost_hr": 2.70,  "vram_gb": 80,  "notes": "8-GPU node, ~$21.60/hr total"},
    "CoreWeave - B200":                     {"provider": "CoreWeave", "gpu": "B200",  "cost_hr": 8.60,  "vram_gb": 180, "notes": "8-GPU node, ~$68.80/hr total"},
    "CoreWeave - GB200 NVL72":              {"provider": "CoreWeave", "gpu": "GB200", "cost_hr": 10.50, "vram_gb": 186, "notes": "4-GPU node, ~$42.00/hr total"},
    "CoreWeave - H100 SXM":                 {"provider": "CoreWeave", "gpu": "H100",  "cost_hr": 6.16,  "vram_gb": 80,  "notes": "8-GPU node, ~$49.24/hr total"},
    "CoreWeave - H200":                     {"provider": "CoreWeave", "gpu": "H200",  "cost_hr": 6.31,  "vram_gb": 141, "notes": "8-GPU node, ~$50.44/hr total"},
    "CoreWeave - L40S":                     {"provider": "CoreWeave", "gpu": "L40S",  "cost_hr": 2.25,  "vram_gb": 48,  "notes": "8-GPU node, ~$18.00/hr total"},
    # ── Crusoe ───────────────────────────────────────────────────────────────
    "Crusoe - A100 PCIe 40GB":              {"provider": "Crusoe",    "gpu": "A100",  "cost_hr": 1.45,  "vram_gb": 40,  "notes": "On-demand"},
    "Crusoe - A100 SXM 80GB":              {"provider": "Crusoe",    "gpu": "A100",  "cost_hr": 1.95,  "vram_gb": 80,  "notes": "On-demand"},
    "Crusoe - H100 HGX":                    {"provider": "Crusoe",    "gpu": "H100",  "cost_hr": 3.90,  "vram_gb": 80,  "notes": "On-demand"},
    "Crusoe - H200 HGX":                    {"provider": "Crusoe",    "gpu": "H200",  "cost_hr": 4.29,  "vram_gb": 141, "notes": "On-demand"},
    "Crusoe - L40S":                        {"provider": "Crusoe",    "gpu": "L40S",  "cost_hr": 1.00,  "vram_gb": 48,  "notes": "On-demand"},
    "Crusoe - MI300X":                      {"provider": "Crusoe",    "gpu": "MI300X", "cost_hr": 3.45, "vram_gb": 192, "notes": "On-demand"},
    # ── FluidStack ───────────────────────────────────────────────────────────
    "FluidStack - A100 SXM 80GB":           {"provider": "FluidStack", "gpu": "A100", "cost_hr": 1.30,  "vram_gb": 80,  "notes": "On-demand"},
    "FluidStack - H100 SXM":                {"provider": "FluidStack", "gpu": "H100", "cost_hr": 2.10,  "vram_gb": 80,  "notes": "On-demand"},
    "FluidStack - H200":                    {"provider": "FluidStack", "gpu": "H200", "cost_hr": 2.30,  "vram_gb": 141, "notes": "On-demand"},
    "FluidStack - L40S":                    {"provider": "FluidStack", "gpu": "L40S", "cost_hr": 1.30,  "vram_gb": 48,  "notes": "On-demand"},
    # ── GCP ──────────────────────────────────────────────────────────────────
    "GCP - A100 40GB - a2-highgpu-1g":      {"provider": "GCP",       "gpu": "A100",  "cost_hr": 3.67,  "vram_gb": 40,  "notes": "1-GPU instance"},
    "GCP - H100 SXM - a3-highgpu-8g":       {"provider": "GCP",       "gpu": "H100",  "cost_hr": 10.98, "vram_gb": 80,  "notes": "8-GPU node, ~$87.83/hr total"},
    "GCP - H200 - a3-ultragpu-8g":          {"provider": "GCP",       "gpu": "H200",  "cost_hr": 10.85, "vram_gb": 141, "notes": "8-GPU node, ~$86.76/hr total"},
    "GCP - L4 - g2-standard-4":             {"provider": "GCP",       "gpu": "L4",    "cost_hr": 0.71,  "vram_gb": 24,  "notes": "1-GPU instance"},
    "GCP - L4 - g2-standard-48":            {"provider": "GCP",       "gpu": "L4",    "cost_hr": 1.00,  "vram_gb": 24,  "notes": "4-GPU node, ~$4.00/hr total"},
    # ── Lambda ───────────────────────────────────────────────────────────────
    "Lambda - A100 SXM 40GB":               {"provider": "Lambda",    "gpu": "A100",  "cost_hr": 1.48,  "vram_gb": 40,  "notes": "1-GPU instance"},
    "Lambda - A100 SXM 80GB":               {"provider": "Lambda",    "gpu": "A100",  "cost_hr": 2.06,  "vram_gb": 80,  "notes": "8-GPU node pricing"},
    "Lambda - B200 SXM":                    {"provider": "Lambda",    "gpu": "B200",  "cost_hr": 5.74,  "vram_gb": 180, "notes": "8-GPU node pricing"},
    "Lambda - H100 PCIe":                   {"provider": "Lambda",    "gpu": "H100",  "cost_hr": 2.86,  "vram_gb": 80,  "notes": "1-GPU instance"},
    "Lambda - H100 SXM":                    {"provider": "Lambda",    "gpu": "H100",  "cost_hr": 3.44,  "vram_gb": 80,  "notes": "8-GPU node pricing"},
    # ── RunPod ───────────────────────────────────────────────────────────────
    "RunPod - A100 PCIe 80GB":              {"provider": "RunPod",    "gpu": "A100",  "cost_hr": 1.19,  "vram_gb": 80,  "notes": "Secure Cloud on-demand"},
    "RunPod - A100 SXM 80GB":               {"provider": "RunPod",    "gpu": "A100",  "cost_hr": 1.39,  "vram_gb": 80,  "notes": "Secure Cloud on-demand"},
    "RunPod - B200":                        {"provider": "RunPod",    "gpu": "B200",  "cost_hr": 5.98,  "vram_gb": 180, "notes": "Secure Cloud on-demand"},
    "RunPod - H100 PCIe":                   {"provider": "RunPod",    "gpu": "H100",  "cost_hr": 1.99,  "vram_gb": 80,  "notes": "Secure Cloud on-demand"},
    "RunPod - H100 SXM":                    {"provider": "RunPod",    "gpu": "H100",  "cost_hr": 2.69,  "vram_gb": 80,  "notes": "Secure Cloud on-demand"},
    "RunPod - H200 SXM":                    {"provider": "RunPod",    "gpu": "H200",  "cost_hr": 3.59,  "vram_gb": 141, "notes": "Secure Cloud on-demand"},
    "RunPod - L4":                          {"provider": "RunPod",    "gpu": "L4",    "cost_hr": 0.39,  "vram_gb": 24,  "notes": "Secure Cloud on-demand"},
    "RunPod - L40S":                        {"provider": "RunPod",    "gpu": "L40S",  "cost_hr": 0.79,  "vram_gb": 48,  "notes": "Secure Cloud on-demand"},
    "RunPod - MI300X":                      {"provider": "RunPod",    "gpu": "MI300X", "cost_hr": 2.99, "vram_gb": 192, "notes": "Community Cloud on-demand"},
    # ── Together AI ──────────────────────────────────────────────────────────
    "Together - B200":                      {"provider": "Together",  "gpu": "B200",  "cost_hr": 7.49,  "vram_gb": 180, "notes": "GPU cluster on-demand"},
    "Together - H100":                      {"provider": "Together",  "gpu": "H100",  "cost_hr": 3.49,  "vram_gb": 80,  "notes": "GPU cluster on-demand"},
    "Together - H200":                      {"provider": "Together",  "gpu": "H200",  "cost_hr": 4.19,  "vram_gb": 141, "notes": "GPU cluster on-demand"},
    # ── Vast.ai ──────────────────────────────────────────────────────────────
    "Vast.ai - A100 SXM 80GB":             {"provider": "Vast.ai",   "gpu": "A100",  "cost_hr": 0.77,  "vram_gb": 80,  "notes": "Marketplace median"},
    "Vast.ai - B200":                       {"provider": "Vast.ai",   "gpu": "B200",  "cost_hr": 2.67,  "vram_gb": 192, "notes": "Marketplace pricing"},
    "Vast.ai - H100 SXM":                  {"provider": "Vast.ai",   "gpu": "H100",  "cost_hr": 1.91,  "vram_gb": 80,  "notes": "Marketplace median"},
    "Vast.ai - H200":                      {"provider": "Vast.ai",   "gpu": "H200",  "cost_hr": 2.52,  "vram_gb": 141, "notes": "Marketplace median"},
    "Vast.ai - L4":                         {"provider": "Vast.ai",   "gpu": "L4",    "cost_hr": 0.34,  "vram_gb": 24,  "notes": "Marketplace median"},
    "Vast.ai - L40S":                       {"provider": "Vast.ai",   "gpu": "L40S",  "cost_hr": 0.50,  "vram_gb": 48,  "notes": "Marketplace median"},
}

GPU_INSTANCES = list(GPU_LIBRARY.keys())


# ═════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def sf(val, default=0):
    """Safe float conversion."""
    try:
        return float(val) if val is not None else default
    except (ValueError, TypeError):
        return default


def fmt_c(val, decimals=2):
    """Format as currency string."""
    return f"${val:,.{decimals}f}"


def fmt_n(val, decimals=0):
    """Format number with commas."""
    return f"{val:,.{decimals}f}"


def fmt_p(val, decimals=1):
    """Format as percentage string."""
    return f"{val * 100:.{decimals}f}%"


def _card(label, value, style="default"):
    """Return an HTML metric card using CSS classes for theme compatibility."""
    return (
        f'<div class="tco-card tco-card-{style}">'
        f'<div class="tco-card-label">{label}</div>'
        f'<div class="tco-card-value">{value}</div></div>'
    )


def _cards_row(*cards):
    """Wrap card HTML strings in a flex row."""
    return f'<div style="display:flex;gap:0.75rem;margin:0.75rem 0;flex-wrap:wrap">{"".join(cards)}</div>'


def get_model_prices(model_name):
    """Return (input_price, output_price) from model library."""
    if model_name and model_name in MODEL_LIBRARY:
        m = MODEL_LIBRARY[model_name]
        if m["input"] is not None:
            return float(m["input"]), float(m["output"])
    return 0.0, 0.0


def get_gpu_price(instance_name):
    """Return hourly cost from GPU library, or leave unchanged for Custom."""
    if instance_name and instance_name in GPU_LIBRARY:
        return float(GPU_LIBRARY[instance_name]["cost_hr"])
    return gr.update()


# ═════════════════════════════════════════════════════════════════════════════
# CALCULATION FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════

def calc_usage(input_tpr, output_tpr, req_day, days_year):
    input_day = input_tpr * req_day
    output_day = output_tpr * req_day
    return {
        "input_day": input_day,
        "output_day": output_day,
        "total_day": input_day + output_day,
        "input_year_M": input_day * days_year / 1e6,
        "output_year_M": output_day * days_year / 1e6,
    }


def calc_api(name, in_price, out_price, in_year_M, out_year_M, req_day, days_year):
    a_in = in_year_M * in_price
    a_out = out_year_M * out_price
    total = a_in + a_out
    total_req = req_day * days_year
    return {
        "name": name,
        "in_price": in_price,
        "out_price": out_price,
        "in_M": in_year_M,
        "out_M": out_year_M,
        "a_in": a_in,
        "a_out": a_out,
        "total": total,
        "monthly": total / 12,
        "per_1k": (total / total_req * 1000) if total_req > 0 else 0,
    }


def calc_smart_routing(providers):
    """60% cheapest, 40% 2nd cheapest among non-zero providers."""
    valid = [p for p in providers if p["total"] > 0]
    if not valid:
        return {"annual": 0, "monthly": 0, "savings": 0}
    valid.sort(key=lambda x: x["total"])
    if len(valid) == 1:
        blended = valid[0]["total"]
    else:
        blended = 0.6 * valid[0]["total"] + 0.4 * valid[1]["total"]
    avg = sum(p["total"] for p in valid) / len(valid)
    return {
        "annual": blended,
        "monthly": blended / 12,
        "savings": 1 - blended / avg if avg > 0 else 0,
    }


def calc_self_hosted(gpu_cost_hr, num_gpus, hours_day, days_year, throughput,
                     utilization, salary, pct_time, setup_cost,
                     total_day, total_year_M):
    gpu = gpu_cost_hr * num_gpus * hours_day * days_year
    eng = salary * pct_time / 100
    setup_a = setup_cost / 3
    sw, net = 2000, 3000
    total = gpu + eng + setup_a + sw + net
    max_tok = throughput * num_gpus * 3600 * hours_day * utilization / 100
    headroom = ((max_tok - total_day) / total_day) if total_day > 0 else float("inf")
    return {
        "gpu": gpu, "eng": eng, "setup_a": setup_a, "sw": sw, "net": net,
        "total": total, "monthly": total / 12,
        "max_tok": max_tok, "headroom": headroom,
        "cost_per_M": total / total_year_M if total_year_M > 0 else 0,
        "setup_cost": setup_cost,
    }


def calc_local(hw_cost, num_dev, lifetime, watts, elec_rate, hours_day, days_year,
               throughput, it_support, total_day, total_year_M):
    hw_a = (hw_cost * num_dev / lifetime) if lifetime > 0 else 0
    elec = watts * num_dev * hours_day / 1000 * days_year * elec_rate
    sw = 500
    total = hw_a + elec + it_support + sw
    max_tok = throughput * 3600 * hours_day * num_dev
    headroom = ((max_tok - total_day) / total_day) if total_day > 0 else float("inf")
    return {
        "hw_a": hw_a, "elec": elec, "it": it_support, "sw": sw,
        "total": total, "monthly": total / 12,
        "max_tok": max_tok, "headroom": headroom,
        "cost_per_M": total / total_year_M if total_year_M > 0 else 0,
    }


# ═════════════════════════════════════════════════════════════════════════════
# CHART FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════

COLORS = {
    "navy": "#1e3a5f", "blue": "#2563eb", "blue_light": "#93c5fd",
    "blue_muted": "#60a5fa", "indigo": "#4f46e5",
    "violet": "#7c3aed", "violet_light": "#a78bfa",
    "amber": "#d97706", "amber_light": "#fbbf24",
    "emerald": "#059669", "emerald_light": "#34d399",
    "slate": "#475569", "slate_light": "#94a3b8",
    "teal": "#0d9488",
}

CHART_LAYOUT = dict(
    template="none",
    height=420,
    margin=dict(t=56, b=44, l=56, r=28),
    font=dict(family="Inter, system-ui, sans-serif", size=13, color="#1e293b"),
    title_font=dict(size=15, family="Inter, system-ui, sans-serif", color="#0f172a"),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
)


def chart_api(providers):
    names = [p["name"] for p in providers]
    fig = go.Figure(data=[
        go.Bar(
            name="Input Cost", x=names,
            y=[p["a_in"] for p in providers],
            marker=dict(color=COLORS["blue_light"], line=dict(width=0)),
            text=[fmt_c(p["a_in"], 0) for p in providers],
            textposition="auto",
            textfont=dict(size=12, color="#1e293b"),
            insidetextfont=dict(size=12, color="#1e293b"),
            outsidetextfont=dict(size=12, color="#1e293b"),
        ),
        go.Bar(
            name="Output Cost", x=names,
            y=[p["a_out"] for p in providers],
            marker=dict(color=COLORS["blue"], line=dict(width=0)),
            text=[fmt_c(p["a_out"], 0) for p in providers],
            textposition="auto",
            textfont=dict(size=12, color="#ffffff"),
            insidetextfont=dict(size=12, color="#ffffff"),
            outsidetextfont=dict(size=12, color="#1e293b"),
        ),
    ])
    fig.update_layout(
        barmode="group",
        title=dict(text="Annual API Costs by Provider"),
        yaxis=dict(title="Annual Cost ($)", gridcolor="rgba(128,128,128,0.2)", zeroline=False),
        xaxis=dict(tickfont=dict(size=12)),
        legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center",
                    font=dict(size=12)),
        bargap=0.25, bargroupgap=0.08,
        **CHART_LAYOUT,
    )
    return fig


def chart_donut(labels, values, title, colors=None):
    if colors is None:
        colors = [COLORS["blue"], COLORS["amber"], COLORS["emerald"],
                  COLORS["violet"], COLORS["slate"]]
    filtered = [(l, v, c) for l, v, c in zip(labels, values, colors) if v > 0]
    if not filtered:
        filtered = [(labels[0], 0.01, colors[0])]
    fl, fv, fc = zip(*filtered)
    fig = go.Figure(data=[go.Pie(
        labels=list(fl), values=list(fv), hole=0.55,
        marker=dict(colors=list(fc), line=dict(color="rgba(128,128,128,0.3)", width=1.5)),
        textinfo="label+percent", textposition="outside",
        textfont=dict(size=12, color="#1e293b"),
        hovertemplate="<b>%{label}</b><br>$%{value:,.0f}<br>%{percent}<extra></extra>",
        pull=[0.02] * len(fl),
    )])
    fig.update_layout(
        title=dict(text=title), showlegend=False,
        **CHART_LAYOUT,
    )
    return fig


def chart_comparison_bars(categories, values, title):
    bar_colors = [COLORS["blue"], COLORS["violet"], COLORS["amber"], COLORS["emerald"]]
    fig = go.Figure(data=[go.Bar(
        x=categories, y=values,
        marker=dict(color=bar_colors, line=dict(width=0)),
        text=[fmt_c(v, 0) if v >= 100 else fmt_c(v, 2) for v in values],
        textposition="outside", textfont=dict(size=12, color="#1e293b"),
    )])
    fig.update_layout(
        title=dict(text=title),
        yaxis=dict(title="Cost ($)", gridcolor="rgba(128,128,128,0.2)", zeroline=False),
        xaxis=dict(tickfont=dict(size=11)),
        bargap=0.35,
        **CHART_LAYOUT,
    )
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# MASTER UPDATE FUNCTION
# ═════════════════════════════════════════════════════════════════════════════

def master_update(
    input_tpr, output_tpr, req_day, days_year,
    model_1, price_in_1, price_out_1,
    model_2, price_in_2, price_out_2,
    model_3, price_in_3, price_out_3,
    model_4_name, price_in_4, price_out_4,
    gpu_cost_hr, num_gpus, gpu_util, gpu_hours, gpu_throughput,
    ml_salary, ml_pct, setup_cost,
    hw_cost, num_dev, watts, elec_rate, hw_life,
    local_hours, local_throughput, it_support,
):
    # ── Safe conversions ──
    input_tpr = sf(input_tpr, 500)
    output_tpr = sf(output_tpr, 200)
    req_day = max(sf(req_day, 10000), 1)
    days_year = max(sf(days_year, 365), 1)
    price_in_1 = sf(price_in_1); price_out_1 = sf(price_out_1)
    price_in_2 = sf(price_in_2); price_out_2 = sf(price_out_2)
    price_in_3 = sf(price_in_3); price_out_3 = sf(price_out_3)
    price_in_4 = sf(price_in_4); price_out_4 = sf(price_out_4)
    gpu_cost_hr = sf(gpu_cost_hr, 2.5)
    num_gpus = sf(num_gpus, 1)
    gpu_util = sf(gpu_util, 70)
    gpu_hours = sf(gpu_hours, 24)
    gpu_throughput = sf(gpu_throughput, 2300)
    ml_salary = sf(ml_salary, 180000)
    ml_pct = sf(ml_pct, 25)
    setup_cost_val = sf(setup_cost, 15000)
    hw_cost = sf(hw_cost, 1999)
    num_dev = sf(num_dev, 1)
    watts = sf(watts, 575)
    elec_rate = sf(elec_rate, 0.12)
    hw_life = max(sf(hw_life, 3), 1)
    local_hours = sf(local_hours, 24)
    local_throughput = sf(local_throughput, 100)
    it_support = sf(it_support, 5000)

    # ── Usage ──
    u = calc_usage(input_tpr, output_tpr, req_day, days_year)
    total_year_M = u["input_year_M"] + u["output_year_M"]

    usage_md = (
        _cards_row(
            _card("Input tokens / day", fmt_n(u["input_day"])),
            _card("Output tokens / day", fmt_n(u["output_day"])),
            _card("Total tokens / day", fmt_n(u["total_day"]), "highlight"),
            _card("Input tokens / year", f"{fmt_n(u['input_year_M'])}M"),
            _card("Output tokens / year", f"{fmt_n(u['output_year_M'])}M"),
        )
    )

    # ── API providers ──
    p1 = calc_api(model_1 or "Provider 1", price_in_1, price_out_1,
                  u["input_year_M"], u["output_year_M"], req_day, days_year)
    p2 = calc_api(model_2 or "Provider 2", price_in_2, price_out_2,
                  u["input_year_M"], u["output_year_M"], req_day, days_year)
    p3 = calc_api(model_3 or "Provider 3", price_in_3, price_out_3,
                  u["input_year_M"], u["output_year_M"], req_day, days_year)
    p4 = calc_api(model_4_name or "Custom Provider", price_in_4, price_out_4,
                  u["input_year_M"], u["output_year_M"], req_day, days_year)
    provs = [p1, p2, p3, p4]

    # Deduplicate provider names for DataFrame columns
    seen = {}
    for p in provs:
        n = p["name"]
        seen[n] = seen.get(n, 0) + 1
        if seen[n] > 1:
            p["col"] = f"{n} ({seen[n]})"
        else:
            p["col"] = n

    # API cost table
    def api_col(p):
        return [
            fmt_c(p["in_price"]), fmt_c(p["out_price"]),
            fmt_n(p["in_M"]), fmt_n(p["out_M"]),
            fmt_c(p["a_in"]), fmt_c(p["a_out"]),
            fmt_c(p["total"]), fmt_c(p["monthly"]),
            fmt_c(p["per_1k"], 3),
        ]

    api_df = pd.DataFrame({
        "Metric": [
            "Input price / 1M tokens ($)", "Output price / 1M tokens ($)",
            "Input tokens/year (M)", "Output tokens/year (M)",
            "Annual input cost ($)", "Annual output cost ($)",
            "Total annual cost ($)", "Monthly cost ($)",
            "Cost per 1K requests ($)",
        ],
        p1["col"]: api_col(p1),
        p2["col"]: api_col(p2),
        p3["col"]: api_col(p3),
        p4["col"]: api_col(p4),
    })

    # Smart routing
    sr = calc_smart_routing(provs)
    smart_md = (
        '<div style="margin-top:1.25rem">'
        '<div style="font-size:0.95rem;font-weight:600;margin-bottom:0.15rem">Smart Routing Scenario</div>'
        '<div style="font-size:0.82rem;opacity:0.6;margin-bottom:0.75rem">'
        'Route 60% to cheapest provider, 40% to 2nd cheapest (across all 4)</div>'
        + _cards_row(
            _card("Blended Annual Cost", fmt_c(sr["annual"]), "highlight"),
            _card("Monthly Blended", fmt_c(sr["monthly"])),
            _card("Savings vs. Average", fmt_p(sr["savings"]), "success"),
        )
        + '</div>'
    )

    api_fig = chart_api(provs)

    # ── Self-hosted GPU ──
    sh = calc_self_hosted(
        gpu_cost_hr, num_gpus, gpu_hours, days_year, gpu_throughput,
        gpu_util, ml_salary, ml_pct, setup_cost_val,
        u["total_day"], total_year_M,
    )

    sh_df = pd.DataFrame({
        "Cost Component": [
            "GPU compute cost", "ML engineer cost (allocated)",
            "Setup cost (amortized over 3 years)",
            "Software licenses (est.)", "Data transfer / networking (est.)",
        ],
        "Annual Cost ($)": [
            fmt_c(sh["gpu"]), fmt_c(sh["eng"]), fmt_c(sh["setup_a"]),
            fmt_c(sh["sw"]), fmt_c(sh["net"]),
        ],
        "Notes": [
            "$/hr x GPUs x hrs/day x days/yr",
            "Salary x % time on project",
            "One-time cost / 3 years",
            "Inference framework, monitoring, etc.",
            "Egress, VPN, load balancing",
        ],
    })

    headroom_str = fmt_p(sh["headroom"]) if sh["headroom"] != float("inf") and sh["headroom"] <= 1000 else "N/A"
    headroom_style = "warning" if sh["headroom"] < 0 else "default"
    headroom_warn = ""
    if sh["headroom"] < 0:
        headroom_warn = (
            '<div style="background:rgba(217,119,6,0.1);border:1px solid rgba(217,119,6,0.3);border-radius:8px;'
            'padding:0.6rem 1rem;color:#d97706;font-size:0.85rem;margin-top:0.5rem;font-weight:500">'
            'Capacity insufficient — add more GPUs.</div>'
        )

    sh_summary = (
        _cards_row(
            _card("Total Annual Cost", fmt_c(sh["total"]), "highlight"),
            _card("Monthly Cost", fmt_c(sh["monthly"])),
            _card("Cost per 1M Tokens", fmt_c(sh["cost_per_M"])),
        )
        + '<div style="font-size:0.95rem;font-weight:600;margin:1rem 0 0.5rem">Capacity Analysis</div>'
        + _cards_row(
            _card("Max Tokens / Day", fmt_n(sh["max_tok"])),
            _card("Your Daily Need", fmt_n(u["total_day"])),
            _card("Capacity Headroom", headroom_str, headroom_style),
        )
        + headroom_warn
    )

    sh_fig = chart_donut(
        ["GPU Compute", "ML Engineer", "Setup (amortized)", "Software", "Networking"],
        [sh["gpu"], sh["eng"], sh["setup_a"], sh["sw"], sh["net"]],
        "Self-Hosted GPU — Cost Breakdown",
    )

    # ── Local / Edge ──
    le = calc_local(
        hw_cost, num_dev, hw_life, watts, elec_rate, local_hours, days_year,
        local_throughput, it_support, u["total_day"], total_year_M,
    )

    le_df = pd.DataFrame({
        "Cost Component": [
            "Hardware (amortized)", "Electricity",
            "IT support / maintenance", "Software / licensing (est.)",
        ],
        "Annual Cost ($)": [
            fmt_c(le["hw_a"]), fmt_c(le["elec"]),
            fmt_c(le["it"]), fmt_c(le["sw"]),
        ],
        "Notes": [
            "Purchase cost x devices / lifetime years",
            "Watts x devices x hrs/day / 1000 x days/yr x $/kWh",
            "Annual support cost",
            "Ollama, llama.cpp, monitoring tools",
        ],
    })

    le_headroom_str = fmt_p(le["headroom"]) if le["headroom"] != float("inf") and le["headroom"] <= 1000 else "N/A"
    le_headroom_style = "warning" if le["headroom"] < 0 else "default"
    le_warn = ""
    if le["headroom"] < 0:
        le_warn = (
            '<div style="background:rgba(217,119,6,0.1);border:1px solid rgba(217,119,6,0.3);border-radius:8px;'
            'padding:0.6rem 1rem;color:#d97706;font-size:0.85rem;margin-top:0.5rem;font-weight:500">'
            'Capacity insufficient — add more devices.</div>'
        )

    le_summary = (
        _cards_row(
            _card("Total Annual Cost", fmt_c(le["total"]), "highlight"),
            _card("Monthly Cost", fmt_c(le["monthly"])),
            _card("Cost per 1M Tokens", fmt_c(le["cost_per_M"])),
        )
        + '<div style="font-size:0.95rem;font-weight:600;margin:1rem 0 0.5rem">Capacity Analysis</div>'
        + _cards_row(
            _card("Max Tokens / Day", fmt_n(le["max_tok"])),
            _card("Your Daily Need", fmt_n(u["total_day"])),
            _card("Capacity Headroom", le_headroom_str, le_headroom_style),
        )
        + le_warn
    )

    le_fig = chart_donut(
        ["Hardware", "Electricity", "IT Support", "Software"],
        [le["hw_a"], le["elec"], le["it"], le["sw"]],
        "Local / Edge — Cost Breakdown",
        [COLORS["emerald"], COLORS["amber"], COLORS["blue"], COLORS["slate"]],
    )

    # ── Comparison ──
    valid_api = [p for p in provs if p["total"] > 0]
    best_api = min(valid_api, key=lambda x: x["total"]) if valid_api else provs[0]
    best_api_annual = best_api["total"]

    options = {
        "API (Best Single)": best_api_annual,
        "API (Smart Routing)": sr["annual"],
        "Self-Hosted GPU": sh["total"],
        "Local / Edge": le["total"],
    }
    monthly_opts = {
        "API (Best Single)": best_api["monthly"],
        "API (Smart Routing)": sr["monthly"],
        "Self-Hosted GPU": sh["monthly"],
        "Local / Edge": le["monthly"],
    }
    cpm = {
        "API (Best Single)": best_api_annual / total_year_M if total_year_M > 0 else 0,
        "API (Smart Routing)": sr["annual"] / total_year_M if total_year_M > 0 else 0,
        "Self-Hosted GPU": sh["cost_per_M"],
        "Local / Edge": le["cost_per_M"],
    }

    comp_df = pd.DataFrame({
        "Metric": [
            "Annual total cost ($)", "Monthly cost ($)", "Cost per 1M tokens ($)",
            "Data leaves your network?", "ML team required?",
            "Scales with volume?", "EU AI Act compliant?", "Time to deploy",
        ],
        "API (Best Single)": [
            fmt_c(best_api_annual), fmt_c(best_api["monthly"]),
            fmt_c(cpm["API (Best Single)"], 2),
            "Yes", "No", "Linear cost increase",
            "Depends on vendor DPA", "Days",
        ],
        "API (Smart Routing)": [
            fmt_c(sr["annual"]), fmt_c(sr["monthly"]),
            fmt_c(cpm["API (Smart Routing)"], 2),
            "Yes", "No", "Linear cost increase",
            "Depends on vendor DPA", "Days",
        ],
        "Self-Hosted GPU": [
            fmt_c(sh["total"]), fmt_c(sh["monthly"]),
            fmt_c(sh["cost_per_M"], 2),
            "No (your cloud VPC)", "Yes", "Fixed cost (to capacity)",
            "Full control", "Weeks",
        ],
        "Local / Edge": [
            fmt_c(le["total"]), fmt_c(le["monthly"]),
            fmt_c(le["cost_per_M"], 2),
            "No (fully local)", "Minimal", "Fixed cost (to capacity)",
            "Full control", "Days to weeks",
        ],
    })

    # Lowest cost option
    lowest_name = min(options, key=options.get)
    lowest_val = options[lowest_name]
    highest_val = max(options.values())
    savings_val = highest_val - lowest_val
    savings_pct = savings_val / highest_val if highest_val > 0 else 0

    # Break-even: self-hosted vs best API
    sh_recurring = sh["gpu"] + sh["eng"] + sh["sw"] + sh["net"]  # excludes setup
    sh_recurring_monthly = sh_recurring / 12
    if sh_recurring_monthly >= best_api["monthly"]:
        breakeven = "API is cheaper"
    else:
        monthly_saving = best_api["monthly"] - sh_recurring_monthly
        if monthly_saving > 0:
            be_months = setup_cost_val / monthly_saving
            breakeven = f"{be_months:.1f} months"
        else:
            breakeven = "N/A"

    comp_summary = (
        # Winner banner
        '<div style="background:linear-gradient(135deg,#0f172a,#1e293b);border-radius:12px;'
        'padding:1.25rem 1.75rem;margin-bottom:0.75rem;border:1px solid #334155">'
        '<div style="display:flex;align-items:center;gap:1.25rem;flex-wrap:wrap">'
        '<div style="flex:1;min-width:200px">'
        '<div style="font-size:0.72rem;text-transform:uppercase;letter-spacing:0.06em;'
        'color:#94a3b8;font-weight:600;margin-bottom:0.2rem">Lowest Cost Option</div>'
        f'<div style="font-size:1.5rem;font-weight:700;color:#f8fafc">{lowest_name}</div>'
        '</div>'
        '<div style="flex:1;min-width:200px">'
        '<div style="font-size:0.72rem;text-transform:uppercase;letter-spacing:0.06em;'
        'color:#94a3b8;font-weight:600;margin-bottom:0.2rem">Best API Provider</div>'
        f'<div style="font-size:1.5rem;font-weight:700;color:#60a5fa">{best_api["name"]}</div>'
        '</div></div></div>'
        # Metric cards
        + _cards_row(
            _card("Annual Savings", fmt_c(savings_val), "success"),
            _card("Savings %", fmt_p(savings_pct), "success"),
            _card("Break-Even (Self-Hosted)", breakeven,
                  "warning" if breakeven == "API is cheaper" else "default"),
        )
    )

    cats = list(options.keys())
    comp_annual_fig = chart_comparison_bars(
        cats, list(options.values()), "Annual Cost Comparison"
    )
    comp_per_M_fig = chart_comparison_bars(
        cats, list(cpm.values()), "Cost per 1M Tokens"
    )

    return (
        usage_md,
        api_df, smart_md, api_fig,
        sh_df, sh_summary, sh_fig,
        le_df, le_summary, le_fig,
        comp_summary, comp_df, comp_annual_fig, comp_per_M_fig,
    )


# ═════════════════════════════════════════════════════════════════════════════
# UI LAYOUT
# ═════════════════════════════════════════════════════════════════════════════

CSS = """
/* ── Base ── */
.gradio-container { max-width: 1280px !important; margin: 0 auto !important; }
footer { display: none !important; }

/* ── Dropdown list: show all models without scrollbar ── */
ul.options { max-height: none !important; }

/* ── Read-only number inputs: look normal, just not editable ── */
input[type="number"]:disabled {
    opacity: 1 !important;
    -webkit-text-fill-color: inherit !important;
    cursor: default !important;
}

/* ── Header banner ── */
.hero-banner {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
    border-radius: 14px;
    padding: 2.25rem 2.75rem;
    margin-bottom: 0.75rem;
    border: 1px solid #334155;
    position: relative;
}
.hero-banner h1 {
    color: #f8fafc !important; font-size: 1.75rem !important;
    font-weight: 700 !important; margin: 0 0 0.35rem 0 !important;
    letter-spacing: -0.03em;
}
.hero-banner .byline {
    color: #94a3b8 !important; font-size: 0.9rem !important;
    margin: 0 0 0.6rem 0 !important; font-weight: 400;
}
.hero-banner .tagline {
    color: #cbd5e1 !important; font-size: 0.925rem !important;
    margin: 0 !important; line-height: 1.5;
}

/* ── Section dividers in inputs tab ── */
.section-label h3 {
    font-size: 0.95rem !important; font-weight: 600 !important;
    border-left: 3px solid #3b82f6;
    padding-left: 0.75rem; margin: 0 !important;
}
.section-label p {
    font-size: 0.82rem !important; opacity: 0.65;
    padding-left: 0.95rem; margin: 0.15rem 0 0 0 !important;
}

/* ── Tab styling ── */
.tab-nav button {
    font-weight: 500 !important; font-size: 0.875rem !important;
    letter-spacing: 0.01em;
}
.tab-nav button.selected {
    font-weight: 600 !important;
}

/* ── Dataframe polish ── */
.dataframe-container table { font-size: 0.875rem; }
.dataframe-container th {
    font-weight: 600 !important; text-transform: uppercase;
    font-size: 0.78rem !important; letter-spacing: 0.03em;
}

/* ── Chart containers (theme-aware Plotly) ── */
.plot-container { border-radius: 10px; overflow: hidden; }
.plot-container,
.plot-container > div,
.plot-container .js-plotly-plot,
.plot-container .plotly,
.plot-container .svg-container,
.plot-container .main-svg {
    background: transparent !important;
    background-color: transparent !important;
}
.plot-container .js-plotly-plot .legendtext,
.plot-container .js-plotly-plot .gtitle,
.plot-container .js-plotly-plot .xtick text,
.plot-container .js-plotly-plot .ytick text,
.plot-container .js-plotly-plot .g-xtitle text,
.plot-container .js-plotly-plot .g-ytitle text {
    fill: #1e293b !important;
}
.plot-container .js-plotly-plot .gridlayer line {
    stroke: rgba(128,128,128,0.2) !important;
}
.plot-container .js-plotly-plot .zerolinelayer line {
    stroke: rgba(128,128,128,0.3) !important;
}

/* ── Metric cards (theme-aware) ── */
.tco-card {
    flex: 1; border-radius: 10px; padding: 1rem 1.25rem;
    text-align: center; min-width: 140px;
    border: 1px solid var(--border-color-primary, rgba(128,128,128,0.2));
    background: var(--background-fill-secondary, rgba(128,128,128,0.06));
}
.tco-card-label {
    font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em;
    opacity: 0.6; margin-bottom: 0.3rem; font-weight: 600;
}
.tco-card-value {
    font-size: 1.4rem; font-weight: 700; line-height: 1.2;
}
/* Card variants */
.tco-card-highlight {
    background: rgba(59,130,246,0.08);
    border-color: rgba(59,130,246,0.3);
}
.tco-card-highlight .tco-card-value { color: #3b82f6; }
.tco-card-highlight .tco-card-label { color: #3b82f6; opacity: 0.8; }

.tco-card-success {
    background: rgba(5,150,105,0.08);
    border-color: rgba(5,150,105,0.3);
}
.tco-card-success .tco-card-value { color: #059669; }
.tco-card-success .tco-card-label { color: #059669; opacity: 0.8; }

.tco-card-warning {
    background: rgba(217,119,6,0.08);
    border-color: rgba(217,119,6,0.3);
}
.tco-card-warning .tco-card-value { color: #d97706; }
.tco-card-warning .tco-card-label { color: #d97706; opacity: 0.8; }
"""


def build_app():
    theme = gr.themes.Default(
        primary_hue="blue",
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    )

    # Force light mode by intercepting matchMedia before Gradio reads it.
    # This prevents Gradio from ever detecting dark mode — no fighting,
    # no MutationObserver loops, no timing issues.
    THEME_HEAD = """
    <script>
    (function() {
        var _mm = window.matchMedia;
        window.matchMedia = function(q) {
            if (q === '(prefers-color-scheme: dark)') {
                return {matches:false, media:q,
                    addEventListener:function(){}, removeEventListener:function(){},
                    addListener:function(){}, removeListener:function(){}};
            }
            return _mm.call(window, q);
        };
        document.addEventListener('DOMContentLoaded', function() {
            if (document.body) document.body.classList.remove('dark');
        });
    })();
    </script>
    """

    # Gradio 6.x moved theme/css/head to launch(); detect version for compat
    gradio_major = int(gr.__version__.split(".")[0])
    blocks_kwargs = {"title": "AI Infrastructure TCO Calculator"}
    if gradio_major < 6:
        blocks_kwargs.update(theme=theme, css=CSS, head=THEME_HEAD)

    with gr.Blocks(**blocks_kwargs) as demo:

        gr.Markdown(
            '<div class="hero-banner">'
            '<h1>AI Infrastructure TCO Calculator</h1>'
            '<p class="byline">By Julien Simon &nbsp;|&nbsp; AI Operating Partner, Fortino Capital &nbsp;|&nbsp; March 2026 Pricing</p>'
            '<p class="tagline">Compare API costs, self-hosted GPU, and local/edge deployment for AI inference workloads.<br>'
            'Fill in your parameters below, then explore the analysis tabs.</p>'
            '</div>'
        )

        with gr.Tabs():

            # ─────────────────── Tab 1: Your Inputs ───────────────────
            with gr.Tab("Your Inputs"):
                gr.Markdown("### Usage Parameters\nDefine your workload volume and operating schedule", elem_classes="section-label")
                with gr.Row():
                    input_tpr = gr.Number(
                        value=500, label="Avg tokens/request (input)",
                        info="Typical prompt length in tokens")
                    output_tpr = gr.Number(
                        value=200, label="Avg tokens/request (output)",
                        info="Typical response length in tokens")
                    req_day = gr.Number(
                        value=10000, label="Requests per day",
                        info="Total API/inference calls per day")
                    days_year = gr.Number(
                        value=365, label="Days of operation / year",
                        info="365 for always-on")

                usage_md = gr.Markdown()

                gr.Markdown("---")
                gr.Markdown("### API Pricing (per 1M tokens)\nSelect models from dropdowns for Providers 1-3. Provider 4 is fully custom.", elem_classes="section-label")

                with gr.Row():
                    with gr.Column():
                        model_1 = gr.Dropdown(
                            choices=API_MODELS, value="Claude Sonnet 4.6",
                            label="Provider 1: Model",
                            )
                        price_in_1 = gr.Number(
                            value=3, label="Provider 1: Input $ / 1M tokens", interactive=False)
                        price_out_1 = gr.Number(
                            value=15, label="Provider 1: Output $ / 1M tokens", interactive=False)
                    with gr.Column():
                        model_2 = gr.Dropdown(
                            choices=API_MODELS, value="GPT-5",
                            label="Provider 2: Model")
                        price_in_2 = gr.Number(
                            value=1.25, label="Provider 2: Input $ / 1M tokens", interactive=False)
                        price_out_2 = gr.Number(
                            value=10, label="Provider 2: Output $ / 1M tokens", interactive=False)
                    with gr.Column():
                        model_3 = gr.Dropdown(
                            choices=API_MODELS, value="Gemini 2.5 Flash",
                            label="Provider 3: Model")
                        price_in_3 = gr.Number(
                            value=0.15, label="Provider 3: Input $ / 1M tokens", interactive=False)
                        price_out_3 = gr.Number(
                            value=0.6, label="Provider 3: Output $ / 1M tokens", interactive=False)
                    with gr.Column():
                        model_4_name = gr.Textbox(
                            value="Custom Provider",
                            label="Provider 4: Name (custom)",
                            info="e.g., Together.ai, Groq, Fireworks")
                        price_in_4 = gr.Number(
                            value=0.5, label="Provider 4: Input $ / 1M tokens")
                        price_out_4 = gr.Number(
                            value=1.5, label="Provider 4: Output $ / 1M tokens")

                gr.Markdown("---")
                gr.Markdown("### Self-Hosted GPU Parameters\nCloud GPU rental with your own inference stack", elem_classes="section-label")
                with gr.Row():
                    gpu_instance = gr.Dropdown(
                        choices=["(Custom)"] + GPU_INSTANCES,
                        value="(Custom)",
                        label="GPU instance preset",
                        info="Select to auto-fill hourly cost, or use Custom"
                    )
                    gpu_cost_hr = gr.Number(
                        value=2.5, label="GPU cost / hour ($)",
                        info="H100: $1.49-$3.90, H200: $2.50-$4.31, B200: $3.75-$5.87")
                    num_gpus = gr.Number(
                        value=1, label="Number of GPUs",
                        info="7B model: 1 GPU. 70B model: 2-4 GPUs")
                    gpu_util = gr.Slider(
                        minimum=0, maximum=100, value=70, step=5,
                        label="GPU utilization (%)",
                        info="60-80% typical")
                    gpu_hours = gr.Number(
                        value=24, label="Hours / day running",
                        info="24 for always-on")
                with gr.Row():
                    gpu_throughput = gr.Number(
                        value=2300, label="Throughput (tokens/sec/GPU)",
                        info="vLLM on H100: ~2300 for 8B model")
                    ml_salary = gr.Number(
                        value=180000, label="ML engineer salary (annual $)",
                        info="Fully loaded cost")
                    ml_pct = gr.Slider(
                        minimum=0, maximum=100, value=25, step=5,
                        label="ML engineer % time on project")
                    setup_cost = gr.Number(
                        value=15000, label="Setup & migration cost ($)",
                        info="One-time: deployment, testing, optimization")

                gr.Markdown("---")
                gr.Markdown("### Local / Edge Parameters\nOn-premises or edge deployment with consumer hardware", elem_classes="section-label")
                with gr.Row():
                    hw_cost = gr.Number(
                        value=1999, label="Hardware purchase cost ($)",
                        info="One-time CapEx. RTX 5090: $1,999")
                    num_dev = gr.Number(
                        value=1, label="Number of devices")
                    watts = gr.Number(
                        value=575, label="Power consumption (W / device)",
                        info="RTX 5090: 575W, M4 Max: ~60W")
                    elec_rate = gr.Number(
                        value=0.12, label="Electricity cost ($ / kWh)",
                        info="Average commercial rate")
                with gr.Row():
                    hw_life = gr.Number(
                        value=3, label="Hardware lifetime (years)")
                    local_hours = gr.Number(
                        value=24, label="Hours / day running")
                    local_throughput = gr.Number(
                        value=100, label="Inference throughput (tok/s)",
                        info="llama.cpp quantized: 50-100 on consumer GPU")
                    it_support = gr.Number(
                        value=5000, label="IT support cost (annual $)",
                        info="Maintenance, updates, monitoring")

            # ─────────────────── Tab 2: API Costs ─────────────────────
            with gr.Tab("API Costs"):
                gr.Markdown("### LLM API Cost Analysis\nCosts for 4 API providers based on your usage inputs", elem_classes="section-label")
                api_table = gr.Dataframe(label="API Cost Comparison", interactive=False)
                smart_md = gr.Markdown()
                api_chart = gr.Plot()

            # ─────────────────── Tab 3: Self-Hosted GPU ───────────────
            with gr.Tab("Self-Hosted GPU"):
                gr.Markdown("### Self-Hosted GPU Cost Analysis\nCloud GPU rental with managed inference infrastructure", elem_classes="section-label")
                sh_table = gr.Dataframe(label="Cost Breakdown", interactive=False)
                sh_summary = gr.Markdown()
                sh_chart = gr.Plot()

            # ─────────────────── Tab 4: Local / Edge ──────────────────
            with gr.Tab("Local / Edge"):
                gr.Markdown("### Local / Edge Deployment Cost Analysis\nOn-premises deployment with consumer or edge hardware", elem_classes="section-label")
                le_table = gr.Dataframe(label="Cost Breakdown", interactive=False)
                le_summary = gr.Markdown()
                le_chart = gr.Plot()

            # ─────────────────── Tab 5: Comparison ────────────────────
            with gr.Tab("Comparison"):
                gr.Markdown("### Side-by-Side Comparison\nAll costs annualized. API (Best Single) = cheapest among your 4 selected providers.", elem_classes="section-label")
                comp_summary = gr.Markdown()
                comp_table = gr.Dataframe(label="Comparison", interactive=False)
                with gr.Row():
                    comp_annual_chart = gr.Plot()
                    comp_per_M_chart = gr.Plot()

            # ─────────────────── Tab 6: Model Library ─────────────────
            with gr.Tab("Model Library"):
                gr.Markdown("### Model Library — March 2026 Pricing\nSources: openai.com/api/pricing, docs.anthropic.com, ai.google.dev/gemini-api/docs/pricing, openrouter.ai", elem_classes="section-label")
                lib_rows = []
                for name, m in MODEL_LIBRARY.items():
                    inp = f"${m['input']}" if m["input"] is not None else "N/A (self-hosted)"
                    out = f"${m['output']}" if m["output"] is not None else "N/A (self-hosted)"
                    lib_rows.append([name, m["provider"], inp, out, m["notes"]])
                lib_df = pd.DataFrame(
                    lib_rows,
                    columns=["Model Name", "Provider", "Input $/M tok",
                             "Output $/M tok", "Notes"],
                )
                gr.Dataframe(value=lib_df, label="Model Library", interactive=False)

            # ─────────────────── Tab 7: GPU Library ─────────────────
            with gr.Tab("GPU Library"):
                gr.Markdown("### GPU Instance Pricing — March 2026\nPer-GPU on-demand hourly rates across major cloud providers", elem_classes="section-label")
                gpu_df = pd.DataFrame([
                    {"Instance": k, "Provider": v["provider"], "GPU": v["gpu"],
                     "$/hr": v["cost_hr"], "VRAM (GB)": v["vram_gb"], "Notes": v["notes"]}
                    for k, v in GPU_LIBRARY.items()
                ])
                gr.Dataframe(value=gpu_df, label="GPU Instance Pricing (March 2026)", interactive=False)
                gr.Markdown("*Sources: aws.amazon.com, cloud.google.com, azure.microsoft.com, coreweave.com, crusoe.ai, fluidstack.io, lambda.ai, runpod.io, together.ai, vast.ai*")

        # ─────────────────── Event Wiring ─────────────────────────────
        all_inputs = [
            input_tpr, output_tpr, req_day, days_year,
            model_1, price_in_1, price_out_1,
            model_2, price_in_2, price_out_2,
            model_3, price_in_3, price_out_3,
            model_4_name, price_in_4, price_out_4,
            gpu_cost_hr, num_gpus, gpu_util, gpu_hours, gpu_throughput,
            ml_salary, ml_pct, setup_cost,
            hw_cost, num_dev, watts, elec_rate, hw_life,
            local_hours, local_throughput, it_support,
        ]

        all_outputs = [
            usage_md,
            api_table, smart_md, api_chart,
            sh_table, sh_summary, sh_chart,
            le_table, le_summary, le_chart,
            comp_summary, comp_table, comp_annual_chart, comp_per_M_chart,
        ]

        # Model dropdowns: auto-populate prices, then recalculate
        for dd, pi, po in [
            (model_1, price_in_1, price_out_1),
            (model_2, price_in_2, price_out_2),
            (model_3, price_in_3, price_out_3),
        ]:
            dd.change(
                fn=get_model_prices, inputs=[dd], outputs=[pi, po],
            ).then(
                fn=master_update, inputs=all_inputs, outputs=all_outputs,
            )

        # GPU dropdown: auto-populate cost, then recalculate
        gpu_instance.change(
            fn=get_gpu_price, inputs=[gpu_instance], outputs=[gpu_cost_hr],
        ).then(
            fn=master_update, inputs=all_inputs, outputs=all_outputs,
        )

        # All non-dropdown inputs trigger recalculation
        change_inputs = [
            input_tpr, output_tpr, req_day, days_year,
            model_4_name, price_in_4, price_out_4,
            gpu_cost_hr, num_gpus, gpu_util, gpu_hours, gpu_throughput,
            ml_salary, ml_pct, setup_cost,
            hw_cost, num_dev, watts, elec_rate, hw_life,
            local_hours, local_throughput, it_support,
        ]
        for inp in change_inputs:
            inp.change(fn=master_update, inputs=all_inputs, outputs=all_outputs)

        # Populate all outputs on initial page load
        demo.load(fn=master_update, inputs=all_inputs, outputs=all_outputs)

    # Gradio 6+ passes theme/css/head via launch() instead of Blocks()
    launch_kwargs = {}
    if gradio_major >= 6:
        launch_kwargs = {"theme": theme, "css": CSS, "head": THEME_HEAD}

    return demo, launch_kwargs


# ═════════════════════════════════════════════════════════════════════════════
# LAUNCH
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    demo, launch_kwargs = build_app()
    demo.launch(**launch_kwargs)
