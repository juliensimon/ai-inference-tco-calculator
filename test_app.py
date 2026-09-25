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
from routers import ROUTER_LIBRARY, ROUTERS, ROUTER_AVAILABILITY
from app import (
    sf, fmt_c, fmt_n, fmt_p,
    get_model_prices, slot_name, access_choices, get_gpu_price, get_gpu_instances,
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
        import models, gpus, routers
        for mod in (models, gpus, routers):
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
        name = next(n for n, m in MODEL_LIBRARY.items() if m["input"] is not None)
        m = MODEL_LIBRARY[name]
        inp, out = get_model_prices(name)
        assert (inp, out) == (float(m["input"]), float(m["output"]))

    def test_unknown_model(self):
        assert get_model_prices("NonExistent") == (0.0, 0.0)

    def test_none_model(self):
        assert get_model_prices(None) == (0.0, 0.0)

    def test_empty_string(self):
        assert get_model_prices("") == (0.0, 0.0)

    def test_unpriced_model_returns_zero(self, monkeypatch):
        """Models with None pricing should return (0.0, 0.0)."""
        # A synthetic model, so the path is exercised even when a refresh
        # prices or removes every real unpriced model.
        monkeypatch.setitem(MODEL_LIBRARY, "Test Unpriced",
                            {"provider": "T", "input": None, "output": None, "notes": ""})
        unpriced = [n for n, m in MODEL_LIBRARY.items() if m["input"] is None]
        for name in unpriced:
            assert get_model_prices(name) == (0.0, 0.0)

    def test_returns_floats(self):
        inp, out = get_model_prices(next(n for n, m in MODEL_LIBRARY.items()
                                         if m["input"] is not None))
        assert isinstance(inp, float)
        assert isinstance(out, float)


class TestGetGpuPrice:
    def test_known_instance(self):
        # Pinned to the library, not a literal: prices change every refresh,
        # but the lookup must always return that instance's cost_hr.
        name = next(iter(GPU_LIBRARY))
        price = get_gpu_price(name)
        assert price == GPU_LIBRARY[name]["cost_hr"]
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
        provider = GPU_PROVIDERS[0]
        result = get_gpu_instances(provider)
        assert "choices" in result
        assert len(result["choices"]) > 0
        assert all(GPU_LIBRARY[name]["provider"] == provider for name in result["choices"])

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

    def test_same_model_via_router_counts_once(self):
        """GPT-6 Sol direct and via a router is one model at two prices, not
        two providers to split traffic between: only its cheapest copy may
        enter the blend and the average."""
        direct = {"total": 10000, "model": "GPT-6 Sol"}
        routed = {"total": 10550, "model": "GPT-6 Sol"}
        other = {"total": 20000, "model": "Claude Sonnet 5"}
        sr = calc_smart_routing([routed, direct, other])
        assert sr["annual"] == pytest.approx(0.6 * 10000 + 0.4 * 20000)
        assert sr["savings"] == pytest.approx(1 - sr["annual"] / 15000)


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

    # A fixed scenario the calculation tests are tuned to, deliberately not
    # the UI defaults (those are tested through _ui_default_args).
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
        args[21] = args[22] = 0   # no sw/net overhead, or $5k/yr alone outweighs the API
        result = master_update(*args)
        comp_summary = result[10]
        # Already past break-even: a plain volume, not a target to reach
        assert "req/day" in comp_summary
        assert "Need" not in comp_summary

    def test_breakeven_with_expensive_gpu(self):
        """GPU costlier than the current API spend, but within capacity at
        break-even, should show 'Need X req/day'."""
        args = list(TestMasterUpdate.DEFAULT_ARGS)
        args[16] = 0.5  # gpu_cost_hr: break-even ~132k req/day (92M tok), inside 139M tok/day capacity
        result = master_update(*args)
        comp_summary = result[10]
        assert "Need" in comp_summary and "req/day" in comp_summary

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


# ═════════════════════════════════════════════════════════════════════════════
# ROUTERS / GATEWAYS
# ═════════════════════════════════════════════════════════════════════════════

class TestRouters:
    """Routers pass provider token prices through and add a fee. The fee must
    be applied exactly once, and choosing Direct must leave every existing
    cost unchanged, or the direct-vs-router comparison is meaningless."""

    # Synthetic, listed everywhere: the weekly refresh rewrites real
    # availability, and a delisting must not fail these fee tests.
    MODEL = "Test Routed"

    @pytest.fixture(autouse=True)
    def _routed_model(self, monkeypatch):
        monkeypatch.setitem(MODEL_LIBRARY, self.MODEL,
                            {"provider": "T", "input": 2, "output": 10, "notes": ""})
        monkeypatch.setitem(ROUTER_AVAILABILITY, self.MODEL, tuple(ROUTERS))
        monkeypatch.setitem(MODEL_LIBRARY, "Test Unpriced",
                            {"provider": "T", "input": None, "output": None, "notes": ""})
        monkeypatch.setitem(ROUTER_AVAILABILITY, "Test Unpriced", tuple(ROUTERS))

    def test_direct_is_the_default_and_free(self):
        assert access_choices(self.MODEL)[0] == "Direct"
        assert ROUTER_LIBRARY["Direct"]["fee_pct"] == 0
        assert get_model_prices(self.MODEL) == get_model_prices(self.MODEL, "Direct")

    @pytest.mark.parametrize("router", ["OpenRouter", "Requesty", "Opper"])
    def test_fee_applied_exactly_once(self, router):
        base_in, base_out = get_model_prices(self.MODEL)
        mult = 1 + ROUTER_LIBRARY[router]["fee_pct"] / 100
        inp, out = get_model_prices(self.MODEL, router)
        assert inp == pytest.approx(base_in * mult)
        assert out == pytest.approx(base_out * mult)

    def test_unknown_router_falls_back_to_direct(self):
        assert get_model_prices(self.MODEL, "NoSuchRouter") == get_model_prices(self.MODEL)

    def test_unpriced_model_stays_zero_through_a_router(self):
        unpriced = [n for n, m in MODEL_LIBRARY.items() if m["input"] is None]
        for name in unpriced:
            assert get_model_prices(name, "OpenRouter") == (0.0, 0.0)

    def test_fees_are_plausible(self):
        for name, r in ROUTER_LIBRARY.items():
            assert 0 <= r["fee_pct"] <= 20, f"{name} fee {r['fee_pct']}% looks wrong"
            for field in ("fee_basis", "byok", "source", "notes"):
                assert field in r, f"{name} missing {field}"

    def test_availability_covers_library_with_known_routers(self):
        assert set(ROUTER_AVAILABILITY) == set(MODEL_LIBRARY), (
            "run: python check_router_catalogs.py --write")
        for name, avail in ROUTER_AVAILABILITY.items():
            assert set(avail) <= set(ROUTERS), f"{name}: unknown router in {avail}"

    def test_slot_name_labels_router_slots(self):
        assert slot_name("GPT-6 Sol", "Direct", "Provider 1") == "GPT-6 Sol"
        assert slot_name("GPT-6 Sol", "Opper", "Provider 1") == "GPT-6 Sol via Opper"
        assert slot_name(None, "Direct", "Provider 2") == "Provider 2"

    def test_direct_vs_router_side_by_side(self):
        """Same model, one slot direct and one via OpenRouter: two distinct
        columns, and the router total is exactly the fee higher."""
        args = list(TestMasterUpdate.DEFAULT_ARGS)
        d_in, d_out = get_model_prices(self.MODEL)
        r_in, r_out = get_model_prices(self.MODEL, "OpenRouter")
        args[4:7] = [self.MODEL, d_in, d_out]
        args[7:10] = [self.MODEL, r_in, r_out]
        api_table = master_update(*args, "Direct", "OpenRouter", "Direct")[1]
        direct_col, router_col = self.MODEL, f"{self.MODEL} via OpenRouter"
        assert direct_col in api_table.columns and router_col in api_table.columns
        row = api_table.index[api_table["Metric"] == "Total annual cost ($)"][0]
        money = lambda v: float(v.replace("$", "").replace(",", ""))
        direct_total = money(api_table.at[row, direct_col])
        router_total = money(api_table.at[row, router_col])
        fee = ROUTER_LIBRARY["OpenRouter"]["fee_pct"]
        assert router_total == pytest.approx(direct_total * (1 + fee / 100), rel=1e-3)

    def test_no_price_for_a_route_that_does_not_exist(self, monkeypatch):
        """A model no router lists must not be priced or labelled as bought
        through one: that purchase path cannot be made."""
        monkeypatch.setitem(MODEL_LIBRARY, "Test Unlisted",
                            {"provider": "T", "input": 1, "output": 2, "notes": ""})
        monkeypatch.setitem(ROUTER_AVAILABILITY, "Test Unlisted", ())
        unlisted = [n for n, m in MODEL_LIBRARY.items()
                    if m["input"] is not None and not ROUTER_AVAILABILITY[n]]
        for name in unlisted:
            assert access_choices(name) == ["Direct"]
            assert get_model_prices(name, "OpenRouter") == get_model_prices(name)
            assert slot_name(name, "OpenRouter", "Provider 1") == name

    def test_router_only_models_have_no_direct_price(self, monkeypatch):
        """Models priced from OpenRouter hosts have no first-party API at
        that price. Offering 'Direct' would show the router's price without
        its fee, as if the fee were avoidable."""
        monkeypatch.setitem(MODEL_LIBRARY, "Test Hosted",
                            {"provider": "T", "input": 1, "output": 2,
                             "notes": "Via OpenRouter.", "direct": False})
        monkeypatch.setitem(ROUTER_AVAILABILITY, "Test Hosted", ("Opper",))
        router_only = [n for n, m in MODEL_LIBRARY.items() if m.get("direct") is False]
        for name in router_only:
            choices = access_choices(name)
            assert "Direct" not in choices and choices, f"{name}: {choices}"
            fee = ROUTER_LIBRARY[choices[0]]["fee_pct"]
            base = MODEL_LIBRARY[name]["input"]
            assert get_model_prices(name, "Direct")[0] == pytest.approx(base * (1 + fee / 100), abs=1e-6)
            assert slot_name(name, "Direct", "Provider 1") == f"{name} via {choices[0]}"

    def test_router_only_flag_matches_notes(self):
        """The flag must follow the price source the notes cite, so a refresh
        that adds an OpenRouter-priced model cannot forget it."""
        for name, m in MODEL_LIBRARY.items():
            via_openrouter = "via openrouter" in m["notes"].lower()
            assert (m.get("direct", True) is False) == via_openrouter, name


# ═════════════════════════════════════════════════════════════════════════════
# FULL-APP REVIEW REGRESSIONS
# ═════════════════════════════════════════════════════════════════════════════

def _money(v):
    return float(v.replace("$", "").replace(",", ""))


def _ui():
    """The built app: its components by label, and its event functions."""
    from app import build_app
    demo, _ = build_app()
    comps = {getattr(b, "label", None): b for b in demo.blocks.values()}
    return demo, comps


def _ui_default_args():
    """master_update's arguments exactly as the page computes them on load."""
    demo, _ = _ui()
    load = next(f for f in demo.fns.values()
                if f.fn is master_update and any(t[1] == "load" for t in f.targets))
    return [c.value for c in load.inputs]


class TestComparisonCapacity:
    """An option that cannot serve the workload is not an option: it must not
    win 'Lowest Cost', and savings must not be measured against it."""

    def _heavy(self):
        args = list(TestMasterUpdate.DEFAULT_ARGS)
        args[2] = 1_000_000   # req/day: 700M tok/day, far beyond GPU and local capacity
        return master_update(*args)

    def test_over_capacity_option_cannot_be_lowest_cost(self):
        summary = self._heavy()[10]
        winner = re.search(r"Lowest Cost Option</div>.*?>([^<]+)</div>", summary).group(1)
        assert winner.startswith("API"), winner

    def test_comparison_table_flags_capacity(self):
        comp_df = self._heavy()[11]
        row = comp_df.index[comp_df["Metric"] == "Serves the workload?"][0]
        assert comp_df.at[row, "Local / Edge"].startswith("No")
        assert comp_df.at[row, "Self-Hosted GPU"].startswith("No")
        assert comp_df.at[row, "API (Best Single)"] == "Yes"


class TestApiLabels:
    def test_same_model_twice_gets_two_bars(self):
        """The table already names duplicates 'X' and 'X (2)'; the chart must
        use the same names or Plotly stacks both on one category."""
        args = list(TestMasterUpdate.DEFAULT_ARGS)
        args[7:10] = args[4:7]   # provider 2 = provider 1
        fig = master_update(*args)[3]
        x = list(fig.data[0].x)
        assert len(set(x)) == len(x), x

    def test_provider_named_metric_keeps_the_label_column(self):
        """'Metric' is the table's row-label column; a provider with that name
        must get its own column instead of overwriting the labels."""
        args = list(TestMasterUpdate.DEFAULT_ARGS)
        args[13] = "Metric"   # provider 4 name
        api_df = master_update(*args)[1]
        assert api_df["Metric"].iloc[0] == "Input price / 1M tokens ($)"
        assert len(api_df.columns) == 5


class TestBreakEvenZeroCapacity:
    def test_zero_utilization_is_over_capacity(self):
        """GPUs at 0% serve nothing, so any break-even volume is unreachable."""
        args = list(TestMasterUpdate.DEFAULT_ARGS)
        args[18] = 0   # gpu_util
        assert "over capacity" in master_update(*args)[10]


class TestHardwareLifetime:
    def test_fractional_lifetime_is_not_rounded_up(self):
        """A 6-month lifetime amortizes the hardware over 6 months: twice
        the yearly cost, not the 1-year figure."""
        args = list(TestMasterUpdate.DEFAULT_ARGS)
        args[27] = 0.5   # hw_life
        le_df = master_update(*args)[7]
        assert _money(le_df.iloc[0, 1]) == pytest.approx(1999 / 0.5)


class TestUiDefaults:
    def test_gpu_count_is_per_gpu(self):
        """Library prices and throughput are per GPU, so the multiplier the
        user enters must be a GPU count, not an instance count (a p5.48xlarge
        is 8 GPUs)."""
        _, comps = _ui()
        assert "Number of GPUs" in comps
        assert "Number of instances" not in comps

    def test_gpu_price_hint_matches_library(self):
        """The hint quotes price ranges; they must come from the library, not
        from an old refresh."""
        _, comps = _ui()
        hint = comps["GPU cost / hour ($)"].info
        for gpu in ("H100", "H200", "B200"):
            prices = [g["cost_hr"] for g in GPU_LIBRARY.values() if g["gpu"] == gpu]
            assert f"{gpu}: ${min(prices):.2f}-${max(prices):.2f}" in hint, hint

    def test_defaults_can_serve_the_default_workload(self):
        """First load must not greet the user with 'Capacity insufficient'."""
        result = master_update(*_ui_default_args())
        assert "insufficient" not in result[5].lower()   # self-hosted summary
        assert "insufficient" not in result[8].lower()   # local summary

    def test_default_best_api_is_a_real_model(self):
        """The headline 'Best API Provider' must not be the unconfigured
        custom placeholder with an invented price."""
        summary = master_update(*_ui_default_args())[10]
        best = re.search(r"Best API Provider</div>.*?>([^<]+)</div>", summary).group(1)
        assert best in MODEL_LIBRARY, best


class TestGpuEventWiring:
    def test_gpu_dropdowns_recalculate_once(self):
        """gpu_cost_hr.change already recalculates. Chaining master_update
        after the provider/instance dropdowns too runs it up to 4 times per
        click, and the runs can finish out of order."""
        from app import get_gpu_instances, get_gpu_price
        demo, comps = _ui()
        fns = list(demo.fns.values())
        by_id = {f._id: f for f in fns}
        dropdown_steps = {f._id for f in fns if f.fn in (get_gpu_instances, get_gpu_price)}
        chained = [f for f in fns if f.fn is master_update
                   and getattr(f, "trigger_after", None) in dropdown_steps]
        assert not chained
        # and the cost box itself still recalculates
        cost_id = comps["GPU cost / hour ($)"]._id
        assert any(f.fn is master_update and (cost_id, "change") in
                   [tuple(t) for t in f.targets] for f in fns)



class TestReviewRound3:
    def _heavy(self, **overrides):
        args = list(TestMasterUpdate.DEFAULT_ARGS)
        args[2] = 1_000_000
        for i, v in overrides.items():
            args[int(i[1:])] = v
        return master_update(*args)

    def test_charts_leave_out_options_that_cannot_serve_the_load(self):
        """The table says Local/Edge is 98% short; its bar must not stand
        next to the API bars as the cheapest option."""
        r = self._heavy()
        for fig in (r[12], r[13]):
            cats = list(fig.data[0].x)
            assert not any("Local" in c or "Self-Hosted" in c for c in cats), cats

    def test_cost_per_token_not_quoted_for_partial_service(self):
        """Fixed cost divided by tokens the hardware cannot serve is a
        fictional unit price."""
        comp_df = self._heavy()[11]
        row = comp_df.index[comp_df["Metric"] == "Cost per 1M tokens ($)"][0]
        assert comp_df.at[row, "Local / Edge"].startswith("N/A")

    def test_break_even_not_reached_when_todays_volume_is_over_capacity(self):
        """Break-even below today's volume means nothing if the GPUs cannot
        serve today's volume."""
        summary = self._heavy(a16=0.5, a21=0, a22=0)[10]
        assert "over capacity" in summary

    def test_dedupe_never_repeats_a_column(self):
        args = list(TestMasterUpdate.DEFAULT_ARGS)
        args[7:10] = args[4:7]
        args[13] = f"{args[4]} (2)"
        r = master_update(*args)
        assert len(r[1].columns) == 5
        x = list(r[3].data[0].x)
        assert len(set(x)) == len(x), x

    def test_unconfigured_custom_provider_is_not_shown_as_free(self):
        """Provider 4 at $0/$0 is unconfigured, not a $0 offer."""
        args = list(TestMasterUpdate.DEFAULT_ARGS)
        args[14] = args[15] = 0
        r = master_update(*args)
        assert "Custom Provider" not in r[1].columns
        assert "Custom Provider" not in list(r[3].data[0].x)

    def test_price_hint_skips_gpus_missing_from_library(self):
        """A refresh that drops a GPU type must shorten the hint, not crash
        app startup."""
        from app import gpu_price_hint
        hint = gpu_price_hint(("H100", "NoSuchGPU"))
        assert hint.startswith("H100: $") and "NoSuchGPU" not in hint

    def test_cleared_device_count_uses_the_ui_default(self):
        args = _ui_default_args()
        args[24] = None   # num_dev cleared
        assert "insufficient" not in master_update(*args)[8].lower()

    def test_zero_lifetime_is_not_a_one_month_write_off(self):
        """0 years is not a lifetime; it must not amortize the hardware 12x
        per year. Invalid input falls back to the default 3 years."""
        args = list(TestMasterUpdate.DEFAULT_ARGS)
        args[27] = 0
        le_df = master_update(*args)[7]
        assert _money(le_df.iloc[0, 1]) == pytest.approx(1999 / 3, abs=0.01)
