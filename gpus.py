"""
GPU Instance Library — per-GPU on-demand hourly pricing for inference
Pricing as of March 2026

Sources: aws.amazon.com, cloud.google.com, azure.microsoft.com,
         coreweave.com, crusoe.ai, fluidstack.io, lambda.ai,
         runpod.io, together.ai, vast.ai
"""

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
    "CoreWeave - B200":                     {"provider": "CoreWeave", "gpu": "B200",  "cost_hr": 8.60,  "vram_gb": 192, "notes": "8-GPU node, ~$68.80/hr total"},
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
    "Lambda - B200 SXM":                    {"provider": "Lambda",    "gpu": "B200",  "cost_hr": 5.74,  "vram_gb": 192, "notes": "8-GPU node pricing"},
    "Lambda - H100 PCIe":                   {"provider": "Lambda",    "gpu": "H100",  "cost_hr": 2.86,  "vram_gb": 80,  "notes": "1-GPU instance"},
    "Lambda - H100 SXM":                    {"provider": "Lambda",    "gpu": "H100",  "cost_hr": 3.44,  "vram_gb": 80,  "notes": "8-GPU node pricing"},
    # ── RunPod ───────────────────────────────────────────────────────────────
    "RunPod - A100 PCIe 80GB":              {"provider": "RunPod",    "gpu": "A100",  "cost_hr": 1.19,  "vram_gb": 80,  "notes": "Secure Cloud on-demand"},
    "RunPod - A100 SXM 80GB":               {"provider": "RunPod",    "gpu": "A100",  "cost_hr": 1.39,  "vram_gb": 80,  "notes": "Secure Cloud on-demand"},
    "RunPod - B200":                        {"provider": "RunPod",    "gpu": "B200",  "cost_hr": 5.98,  "vram_gb": 192, "notes": "Secure Cloud on-demand"},
    "RunPod - H100 PCIe":                   {"provider": "RunPod",    "gpu": "H100",  "cost_hr": 1.99,  "vram_gb": 80,  "notes": "Secure Cloud on-demand"},
    "RunPod - H100 SXM":                    {"provider": "RunPod",    "gpu": "H100",  "cost_hr": 2.69,  "vram_gb": 80,  "notes": "Secure Cloud on-demand"},
    "RunPod - H200 SXM":                    {"provider": "RunPod",    "gpu": "H200",  "cost_hr": 3.59,  "vram_gb": 141, "notes": "Secure Cloud on-demand"},
    "RunPod - L4":                          {"provider": "RunPod",    "gpu": "L4",    "cost_hr": 0.39,  "vram_gb": 24,  "notes": "Secure Cloud on-demand"},
    "RunPod - L40S":                        {"provider": "RunPod",    "gpu": "L40S",  "cost_hr": 0.79,  "vram_gb": 48,  "notes": "Secure Cloud on-demand"},
    "RunPod - MI300X":                      {"provider": "RunPod",    "gpu": "MI300X", "cost_hr": 2.99, "vram_gb": 192, "notes": "Community Cloud on-demand"},
    # ── Together AI ──────────────────────────────────────────────────────────
    "Together - B200":                      {"provider": "Together",  "gpu": "B200",  "cost_hr": 7.49,  "vram_gb": 192, "notes": "GPU cluster on-demand"},
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

GPU_PROVIDERS = sorted(set(v["provider"] for v in GPU_LIBRARY.values()))
