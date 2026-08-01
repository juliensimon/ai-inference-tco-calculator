"""
Unit tests for the AI Inference TCO Calculator.
Tests calculation functions, helpers, data integrity, and master_update.
"""

import pytest
import math
import re
import pathlib
from models import MODEL_LIBRARY, API_MODELS
from gpus import GPU_LIBRARY, GPU_PROVIDERS
from app import (
    sf, fmt_c, fmt_n, fmt_p,
    get_model_prices, get_gpu_price, get_gpu_instances,
    calc_usage, calc_api, calc_smart_routing,
    calc_self_hosted, calc_local, master_update,
    DEFAULT_MODELS, PRICING_DATE,
)

_MONTHS = ("January|February|March|April|May|June|July"
           "|August|September|October|November|December")


class TestPricingDate:
    """The pricing date is shown in four places in the UI. It once drifted:
    the data-file docstrings said July 31 while the banner and both library
    tabs still said July 9, so the deployed Space advertised stale pricing."""

    def test_ui_dates_all_come_from_the_constant(self):
        src = (pathlib.Path(__file__).parent / "app.py").read_text()
        found = set(re.findall(rf"(?:{_MONTHS})\s+\d{{1,2}},\s+\d{{4}}", src))
        assert found <= {PRICING_DATE}, (
            f"hardcoded date(s) in app.py: {sorted(found - {PRICING_DATE})} — "
            f"use PRICING_DATE instead")

    def test_data_files_agree_with_the_constant(self):
        import models, gpus
        for mod in (models, gpus):
            assert f"Pricing as of {PRICING_DATE}" in mod.__doc__, (
                f"{mod.__name__} docstring disagrees with PRICING_DATE")


# ═════════════════════════════════════════════════════════════════════════════
# HELPER TESTS
# ═════════════════════════════════════════════════════════════════════════════

class TestSf:
    def test_normal_float(self):
        assert sf(3.14) == 3.14

    def test_int(self):
        assert sf(5) == 5.0

    def test_string_number(self):
        assert sf("2.5") == 2.5

    def test_none_returns_default(self):
        assert sf(None) == 0
        assert sf(None, 42) == 42

    def test_invalid_string_returns_default(self):
        assert sf("abc") == 0
        assert sf("abc", 99) == 99

    def test_empty_string_returns_default(self):
        assert sf("") == 0

    def test_zero(self):
        assert sf(0) == 0.0

    def test_negative(self):
        assert sf(-5.5) == -5.5


class TestFormatters:
    def test_fmt_c_basic(self):
        assert fmt_c(1234.5) == "$1,234.50"

    def test_fmt_c_zero(self):
        assert fmt_c(0) == "$0.00"

    def test_fmt_c_custom_decimals(self):
        assert fmt_c(1.2345, 3) == "$1.234"

    def test_fmt_c_large(self):
        assert fmt_c(1000000) == "$1,000,000.00"

    def test_fmt_n_basic(self):
        assert fmt_n(1234567) == "1,234,567"

    def test_fmt_n_with_decimals(self):
        assert fmt_n(1234.5678, 2) == "1,234.57"

    def test_fmt_p_basic(self):
        assert fmt_p(0.5) == "50.0%"

    def test_fmt_p_zero(self):
        assert fmt_p(0) == "0.0%"

    def test_fmt_p_over_one(self):
        assert fmt_p(1.5) == "150.0%"

    def test_fmt_p_negative(self):
        assert fmt_p(-0.1) == "-10.0%"


# ═════════════════════════════════════════════════════════════════════════════
# MODEL / GPU LIBRARY DATA INTEGRITY
# ═════════════════════════════════════════════════════════════════════════════

class TestModelLibrary:
    def test_not_empty(self):
        assert len(MODEL_LIBRARY) > 0

    def test_api_models_subset(self):
        """API_MODELS should only contain models with non-None pricing."""
        for name in API_MODELS:
            m = MODEL_LIBRARY[name]
            assert m["input"] is not None
            assert m["output"] is not None

    def test_default_models_are_selectable(self):
        """A default naming a retired/renamed model silently shows $0.00 prices,
        so every default must still be a priced entry in the dropdown."""
        for name in DEFAULT_MODELS:
            assert name in API_MODELS, f"default {name!r} is not a selectable API model"
            assert get_model_prices(name) != (0.0, 0.0), f"default {name!r} priced at zero"

    def test_all_models_have_required_fields(self):
        for name, m in MODEL_LIBRARY.items():
            assert "provider" in m, f"{name} missing provider"
            assert "input" in m, f"{name} missing input"
            assert "output" in m, f"{name} missing output"
            assert "notes" in m, f"{name} missing notes"

    def test_prices_are_non_negative(self):
        for name, m in MODEL_LIBRARY.items():
            if m["input"] is not None:
                assert m["input"] >= 0, f"{name} has negative input price"
                assert m["output"] >= 0, f"{name} has negative output price"

    def test_unpriced_models_excluded_from_api(self):
        """Models with None pricing should not appear in API_MODELS. None means
        "no published per-token price" — self-hosted, subscription-only or
        invite-only alike — not "self-hosted"."""
        for name, m in MODEL_LIBRARY.items():
            if m["input"] is None:
                assert name not in API_MODELS, f"{name} should not be in API_MODELS"

    def test_library_tab_does_not_call_unpriced_models_self_hosted(self):
        """The Model Library once rendered every None-priced model as
        "N/A (self-hosted)". That held only while the sole unpriced entry was
        self-hosted; it became false the moment a subscription-only model
        (Qwen3.8 Max Preview) was the only one left."""
        src = (pathlib.Path(__file__).parent / "app.py").read_text()
        assert "N/A (self-hosted)" not in src, (
            "unpriced models are labelled self-hosted, but None only means "
            "there is no published per-token price")


class TestGPULibrary:
    def test_not_empty(self):
        assert len(GPU_LIBRARY) > 0

    def test_providers_not_empty(self):
        assert len(GPU_PROVIDERS) > 0

    def test_all_gpus_have_required_fields(self):
        for name, g in GPU_LIBRARY.items():
            assert "provider" in g, f"{name} missing provider"
            assert "gpu" in g, f"{name} missing gpu"
            assert "cost_hr" in g, f"{name} missing cost_hr"
            assert "vram_gb" in g, f"{name} missing vram_gb"
            assert "notes" in g, f"{name} missing notes"

    def test_prices_positive(self):
        for name, g in GPU_LIBRARY.items():
            assert g["cost_hr"] > 0, f"{name} has non-positive cost"

    def test_vram_positive(self):
        for name, g in GPU_LIBRARY.items():
            assert g["vram_gb"] > 0, f"{name} has non-positive VRAM"

    def test_providers_match_library(self):
        """GPU_PROVIDERS should exactly match providers in GPU_LIBRARY."""
        actual = sorted(set(v["provider"] for v in GPU_LIBRARY.values()))
        assert GPU_PROVIDERS == actual

    def test_b200_vram_consistency(self):
        """All B200 GPUs should have 192GB VRAM."""
        for name, g in GPU_LIBRARY.items():
            if g["gpu"] == "B200":
                assert g["vram_gb"] == 192, f"{name} B200 has {g['vram_gb']}GB, expected 192GB"

    def test_key_format(self):
        """All keys should follow 'Provider - GPU' or 'Provider - GPU - instance' format."""
        for name in GPU_LIBRARY:
            parts = name.split(" - ")
            assert len(parts) >= 2, f"{name} doesn't follow 'Provider - GPU' format"

    def test_key_prefix_matches_provider(self):
        """The key's provider prefix must match the provider field, or the
        instance is silently filed under the wrong GPU-provider dropdown."""
        for name, g in GPU_LIBRARY.items():
            assert name.split(" - ")[0] == g["provider"], (
                f"{name} has provider {g['provider']!r}")


# ═════════════════════════════════════════════════════════════════════════════
# LOOKUP HELPERS
# ═════════════════════════════════════════════════════════════════════════════

class TestGetModelPrices:
    def test_known_model(self):
        # Pinned to the library, not literals: prices change every refresh,
        # but the lookup must always return that model's input/output pair.
        m = MODEL_LIBRARY["Claude Sonnet 4.6"]
        inp, out = get_model_prices("Claude Sonnet 4.6")
        assert (inp, out) == (float(m["input"]), float(m["output"]))

    def test_unknown_model(self):
        assert get_model_prices("NonExistent") == (0.0, 0.0)

    def test_none_model(self):
        assert get_model_prices(None) == (0.0, 0.0)

    def test_empty_string(self):
        assert get_model_prices("") == (0.0, 0.0)

    def test_unpriced_model_returns_zero(self):
        """Models with None pricing should return (0.0, 0.0)."""
        # Pinned to the library, not a literal: unpriced models come and go.
        unpriced = [n for n, m in MODEL_LIBRARY.items() if m["input"] is None]
        assert unpriced, "no unpriced model in the library to exercise this path"
        for name in unpriced:
            assert get_model_prices(name) == (0.0, 0.0)

    def test_returns_floats(self):
        inp, out = get_model_prices("GPT-5")
        assert isinstance(inp, float)
        assert isinstance(out, float)


class TestGetGpuPrice:
    def test_known_instance(self):
        # Pinned to the library, not a literal: prices change every refresh,
        # but the lookup must always return that instance's cost_hr.
        price = get_gpu_price("RunPod - H100 SXM")
        assert price == GPU_LIBRARY["RunPod - H100 SXM"]["cost_hr"]
        assert isinstance(price, (int, float))

    def test_unknown_instance(self):
        # Should return gr.update() for unknown
        result = get_gpu_price("NonExistent")
        assert not isinstance(result, (int, float))

    def test_custom(self):
        result = get_gpu_price("(Custom)")
        assert not isinstance(result, (int, float))

    def test_none(self):
        result = get_gpu_price(None)
        assert not isinstance(result, (int, float))


class TestGetGpuInstances:
    def test_known_provider(self):
        result = get_gpu_instances("RunPod")
        assert "choices" in result
        assert len(result["choices"]) > 0
        assert all("RunPod" in name for name in result["choices"])

    def test_custom_provider(self):
        result = get_gpu_instances("(Custom)")
        assert result["choices"] == ["(Custom)"]

    def test_none_provider(self):
        result = get_gpu_instances(None)
        assert result["choices"] == ["(Custom)"]

    def test_all_providers_have_instances(self):
        for provider in GPU_PROVIDERS:
            result = get_gpu_instances(provider)
            assert len(result["choices"]) > 0, f"{provider} has no instances"


# ═════════════════════════════════════════════════════════════════════════════
# CALCULATION FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════

class TestCalcUsage:
    def test_basic(self):
        u = calc_usage(500, 200, 10000, 365)
        assert u["input_day"] == 5_000_000
        assert u["output_day"] == 2_000_000
        assert u["total_day"] == 7_000_000
        assert u["input_year_M"] == pytest.approx(1825.0)
        assert u["output_year_M"] == pytest.approx(730.0)

    def test_zero_requests(self):
        u = calc_usage(500, 200, 0, 365)
        assert u["total_day"] == 0
        assert u["input_year_M"] == 0
        assert u["output_year_M"] == 0

    def test_one_request(self):
        u = calc_usage(100, 50, 1, 365)
        assert u["input_day"] == 100
        assert u["output_day"] == 50
        assert u["total_day"] == 150


class TestCalcApi:
    def test_basic(self):
        r = calc_api("Test", 3.0, 15.0, 1825.0, 730.0, 10000, 365)
        assert r["name"] == "Test"
        assert r["a_in"] == pytest.approx(5475.0)   # 1825 * 3
        assert r["a_out"] == pytest.approx(10950.0)  # 730 * 15
        assert r["total"] == pytest.approx(16425.0)
        assert r["monthly"] == pytest.approx(16425.0 / 12)

    def test_zero_prices(self):
        r = calc_api("Free", 0, 0, 1825.0, 730.0, 10000, 365)
        assert r["total"] == 0
        assert r["monthly"] == 0
        assert r["per_1k"] == 0

    def test_per_1k_calculation(self):
        r = calc_api("Test", 3.0, 15.0, 1825.0, 730.0, 10000, 365)
        expected_per_1k = r["total"] / (10000 * 365) * 1000
        assert r["per_1k"] == pytest.approx(expected_per_1k)

    def test_zero_requests(self):
        r = calc_api("Test", 3.0, 15.0, 0, 0, 0, 365)
        assert r["per_1k"] == 0


class TestCalcSmartRouting:
    def test_two_providers(self):
        p1 = {"total": 10000}
        p2 = {"total": 20000}
        sr = calc_smart_routing([p1, p2])
        assert sr["annual"] == pytest.approx(0.6 * 10000 + 0.4 * 20000)
        assert sr["monthly"] == pytest.approx(sr["annual"] / 12)

    def test_single_provider(self):
        p1 = {"total": 10000}
        sr = calc_smart_routing([p1])
        assert sr["annual"] == 10000

    def test_filters_zero_cost(self):
        p1 = {"total": 10000}
        p2 = {"total": 0}
        sr = calc_smart_routing([p1, p2])
        assert sr["annual"] == 10000  # Only p1 counts

    def test_all_zero(self):
        p1 = {"total": 0}
        p2 = {"total": 0}
        sr = calc_smart_routing([p1, p2])
        assert sr["annual"] == 0

    def test_empty_list(self):
        sr = calc_smart_routing([])
        assert sr["annual"] == 0

    def test_savings_positive(self):
        """Blended should always be <= average, so savings >= 0."""
        p1 = {"total": 5000}
        p2 = {"total": 15000}
        p3 = {"total": 25000}
        sr = calc_smart_routing([p1, p2, p3])
        assert sr["savings"] >= 0

    def test_four_providers_uses_top_two(self):
        """Smart routing should use only the 2 cheapest, not all 4."""
        p1 = {"total": 5000}
        p2 = {"total": 10000}
        p3 = {"total": 50000}
        p4 = {"total": 100000}
        sr = calc_smart_routing([p1, p2, p3, p4])
        # Blended = 0.6 * 5000 + 0.4 * 10000 = 7000
        assert sr["annual"] == pytest.approx(7000)


class TestCalcSelfHosted:
    def test_basic(self):
        sh = calc_self_hosted(
            gpu_cost_hr=2.69, num_gpus=1, hours_day=24, days_year=365,
            throughput=2300, utilization=70,
            sw_cost=2000, net_cost=3000,
            total_day=7_000_000, total_year_M=2555.0,
        )
        expected_gpu = 2.69 * 1 * 24 * 365
        assert sh["gpu"] == pytest.approx(expected_gpu)
        assert sh["total"] == pytest.approx(expected_gpu + 2000 + 3000)
        assert sh["monthly"] == pytest.approx(sh["total"] / 12)

    def test_cost_per_M(self):
        sh = calc_self_hosted(
            gpu_cost_hr=2.69, num_gpus=1, hours_day=24, days_year=365,
            throughput=2300, utilization=70,
            sw_cost=0, net_cost=0,
            total_day=7_000_000, total_year_M=2555.0,
        )
        assert sh["cost_per_M"] == pytest.approx(sh["total"] / 2555.0)

    def test_headroom_positive(self):
        sh = calc_self_hosted(
            gpu_cost_hr=2.69, num_gpus=1, hours_day=24, days_year=365,
            throughput=2300, utilization=70,
            sw_cost=0, net_cost=0,
            total_day=1000, total_year_M=0.365,
        )
        assert sh["headroom"] > 0

    def test_headroom_negative(self):
        """Headroom should be negative when demand exceeds capacity."""
        sh = calc_self_hosted(
            gpu_cost_hr=2.69, num_gpus=1, hours_day=1, days_year=365,
            throughput=1, utilization=10,
            sw_cost=0, net_cost=0,
            total_day=999_999_999, total_year_M=999.0,
        )
        assert sh["headroom"] < 0

    def test_zero_tokens(self):
        sh = calc_self_hosted(
            gpu_cost_hr=2.69, num_gpus=1, hours_day=24, days_year=365,
            throughput=2300, utilization=70,
            sw_cost=0, net_cost=0,
            total_day=0, total_year_M=0,
        )
        assert sh["headroom"] == float("inf")
        assert sh["cost_per_M"] == 0

    def test_max_tokens_calculation(self):
        sh = calc_self_hosted(
            gpu_cost_hr=2.69, num_gpus=2, hours_day=24, days_year=365,
            throughput=2300, utilization=100,
            sw_cost=0, net_cost=0,
            total_day=1000, total_year_M=0.365,
        )
        expected_max = 2300 * 2 * 3600 * 24 * 100 / 100
        assert sh["max_tok"] == pytest.approx(expected_max)


class TestCalcLocal:
    def test_basic(self):
        le = calc_local(
            hw_cost=1999, num_dev=1, lifetime=3,
            watts=575, elec_rate=0.12,
            hours_day=24, days_year=365,
            throughput=100, utilization=70, it_support=5000, sw_cost=500,
            total_day=7_000_000, total_year_M=2555.0,
        )
        expected_hw = 1999 / 3
        expected_elec = 575 * 1 * 24 / 1000 * 365 * 0.12
        assert le["hw_a"] == pytest.approx(expected_hw)
        assert le["elec"] == pytest.approx(expected_elec)
        assert le["sw"] == 500
        assert le["total"] == pytest.approx(expected_hw + expected_elec + 5000 + 500)

    def test_zero_lifetime(self):
        le = calc_local(
            hw_cost=1999, num_dev=1, lifetime=0,
            watts=575, elec_rate=0.12,
            hours_day=24, days_year=365,
            throughput=100, utilization=70, it_support=5000, sw_cost=500,
            total_day=1000, total_year_M=0.365,
        )
        assert le["hw_a"] == 0

    def test_capacity_is_derated_by_utilization(self):
        """Local capacity must be derated like self-hosted, otherwise the two
        Capacity Headroom cards are computed on different bases and the
        comparison silently favours local hardware."""
        kwargs = dict(
            hw_cost=1999, num_dev=2, lifetime=3, watts=575, elec_rate=0.12,
            hours_day=24, days_year=365, throughput=100, it_support=5000,
            sw_cost=500, total_day=1000, total_year_M=0.365,
        )
        full = calc_local(utilization=100, **kwargs)
        half = calc_local(utilization=50, **kwargs)
        assert full["max_tok"] == pytest.approx(100 * 2 * 3600 * 24)
        assert half["max_tok"] == pytest.approx(full["max_tok"] / 2)

    def test_software_cost_is_caller_supplied(self):
        """Software cost is a user input, not a hardcoded $500."""
        kwargs = dict(
            hw_cost=0, num_dev=1, lifetime=3, watts=0, elec_rate=0,
            hours_day=24, days_year=365, throughput=100, utilization=70,
            it_support=0, total_day=1000, total_year_M=0.365,
        )
        assert calc_local(sw_cost=0, **kwargs)["total"] == pytest.approx(0)
        assert calc_local(sw_cost=1234, **kwargs)["total"] == pytest.approx(1234)


# ═════════════════════════════════════════════════════════════════════════════
# MASTER UPDATE (INTEGRATION TESTS)
# ═════════════════════════════════════════════════════════════════════════════

class TestMasterUpdate:
    """Integration tests for master_update with default-like values."""

    DEFAULT_ARGS = (
        500, 200, 10000, 365,                        # usage
        "Claude Sonnet 4.6", 3, 15,                  # provider 1
        "GPT-5", 1.25, 10,                           # provider 2
        "Gemini 2.5 Flash", 0.15, 0.6,               # provider 3
        "Custom Provider", 0.5, 1.5,                  # provider 4
        2.5, 1, 70, 24, 2300,                         # GPU params
        2000, 3000,                                    # sw/net costs
        1999, 1, 575, 0.12, 3,                        # local hw params
        70, 24, 100, 5000, 500,                        # local runtime params
    )

    def test_returns_correct_number_of_outputs(self):
        result = master_update(*self.DEFAULT_ARGS)
        assert len(result) == 14  # usage_md + 4 api + 3 sh + 3 le + 3 comp

    def test_returns_no_none(self):
        result = master_update(*self.DEFAULT_ARGS)
        for i, r in enumerate(result):
            assert r is not None, f"Output {i} is None"

    def test_with_zero_requests(self):
        args = list(self.DEFAULT_ARGS)
        args[2] = 0  # req_day = 0 -> gets clamped to 1
        result = master_update(*args)
        assert len(result) == 14

    def test_with_none_values(self):
        """Should handle None inputs gracefully via sf()."""
        args = list(self.DEFAULT_ARGS)
        args[0] = None  # input_tpr
        args[1] = None  # output_tpr
        result = master_update(*args)
        assert len(result) == 14

    def test_with_string_numbers(self):
        """Should handle string inputs via sf()."""
        args = list(self.DEFAULT_ARGS)
        args[0] = "500"
        args[1] = "200"
        result = master_update(*args)
        assert len(result) == 14

    def test_api_costs_are_dataframe(self):
        result = master_update(*self.DEFAULT_ARGS)
        api_table = result[1]  # api_df
        import pandas as pd
        assert isinstance(api_table, pd.DataFrame)

    def test_provider_1_appears_in_table(self):
        result = master_update(*self.DEFAULT_ARGS)
        api_table = result[1]
        assert "Claude Sonnet 4.6" in api_table.columns

    def test_comp_table_has_four_options(self):
        result = master_update(*self.DEFAULT_ARGS)
        comp_table = result[11]  # comp_df
        # Should have API (Best Single), API (Smart Routing), Self-Hosted GPU, Local / Edge
        assert "API (Best Single)" in comp_table.columns
        assert "API (Smart Routing)" in comp_table.columns
        assert "Self-Hosted GPU" in comp_table.columns
        assert "Local / Edge" in comp_table.columns

    def test_all_zero_prices(self):
        """All API prices zero should not crash."""
        args = list(self.DEFAULT_ARGS)
        args[5] = 0; args[6] = 0    # provider 1
        args[8] = 0; args[9] = 0    # provider 2
        args[11] = 0; args[12] = 0  # provider 3
        args[14] = 0; args[15] = 0  # provider 4
        result = master_update(*args)
        assert len(result) == 14

    def test_extreme_values(self):
        """Very large values should not crash."""
        args = list(self.DEFAULT_ARGS)
        args[2] = 10_000_000  # 10M req/day
        result = master_update(*args)
        assert len(result) == 14

    def test_duplicate_model_names(self):
        """Duplicate provider names should be deduplicated in columns."""
        args = list(self.DEFAULT_ARGS)
        args[4] = "GPT-5"   # provider 1
        args[7] = "GPT-5"   # provider 2
        result = master_update(*args)
        api_table = result[1]
        # Should have "GPT-5" and "GPT-5 (2)" as columns
        cols = list(api_table.columns)
        assert "GPT-5" in cols
        assert "GPT-5 (2)" in cols


# ═════════════════════════════════════════════════════════════════════════════
# BREAK-EVEN ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════

class TestBreakEven:
    """Test break-even logic within master_update output."""

    def test_breakeven_in_comparison_summary(self):
        """Comparison summary should contain break-even info."""
        result = master_update(*TestMasterUpdate.DEFAULT_ARGS)
        comp_summary = result[10]  # comp_summary markdown
        assert "Break-Even" in comp_summary

    def test_breakeven_with_cheap_gpu(self):
        """Very cheap GPU should show break-even below current volume."""
        args = list(TestMasterUpdate.DEFAULT_ARGS)
        args[16] = 0.01  # gpu_cost_hr = $0.01 (unrealistically cheap)
        result = master_update(*args)
        comp_summary = result[10]
        # Should show a number, not "Need"
        assert "req/day" in comp_summary

    def test_breakeven_with_expensive_gpu(self):
        """Very expensive GPU should show 'Need X req/day'."""
        args = list(TestMasterUpdate.DEFAULT_ARGS)
        args[16] = 100  # gpu_cost_hr = $100
        result = master_update(*args)
        comp_summary = result[10]
        assert "req/day" in comp_summary

    def test_breakeven_flags_volume_the_gpus_cannot_serve(self):
        """Break-even is pure cost arithmetic, so it can exceed what the
        configured GPUs can actually serve. That has to be said, not presented
        as a reachable target."""
        args = list(TestMasterUpdate.DEFAULT_ARGS)
        args[16] = 100   # gpu_cost_hr = $100 — pushes break-even far out
        args[20] = 1     # gpu_throughput = 1 tok/s — almost no capacity
        result = master_update(*args)
        assert "over capacity" in result[10]

    def test_breakeven_silent_when_capacity_suffices(self):
        """The capacity caveat must not fire when the GPUs can serve it."""
        args = list(TestMasterUpdate.DEFAULT_ARGS)
        args[16] = 0.01      # cheap GPU — break-even well below current volume
        args[20] = 1_000_000  # ample throughput
        result = master_update(*args)
        assert "over capacity" not in result[10]
