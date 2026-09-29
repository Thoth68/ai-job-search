"""Offline tests for model_comparison.py - no API calls, no cost.

Run with: python -m pytest tests/test_model_comparison.py
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import model_comparison as mc  # noqa: E402


def test_pricing_matches_guide():
    assert (mc.MODELS["claude-opus-5-5"]["input"], mc.MODELS["claude-opus-5-5"]["output"]) == (4.0, 20.0)
    assert (mc.MODELS["claude-opus-5"]["input"], mc.MODELS["claude-opus-5"]["output"]) == (5.0, 25.0)
    assert (mc.MODELS["claude-fable-5-1"]["input"], mc.MODELS["claude-fable-5-1"]["output"]) == (10.0, 50.0)


def test_opus_5_5_is_20_percent_cheaper_than_opus_5():
    assert mc.price_ratio("claude-opus-5-5", "claude-opus-5") == pytest.approx(-0.20)


def test_opus_5_5_is_60_percent_cheaper_than_fable_5_1():
    assert mc.price_ratio("claude-opus-5-5", "claude-fable-5-1") == pytest.approx(-0.60)


def test_output_input_ratio_is_5x_for_all_models():
    for p in mc.MODELS.values():
        assert p["output"] / p["input"] == 5


def test_default_effort_levels():
    assert mc.MODELS["claude-opus-5-5"]["default_effort"] == "medium"
    assert mc.MODELS["claude-opus-5"]["default_effort"] == "high"
    assert mc.MODELS["claude-fable-5-1"]["default_effort"] == "high"


def test_cost_usd():
    # 1M input + 1M output on Opus 5.5 = $4 + $20
    assert mc.cost_usd("claude-opus-5-5", 1_000_000, 1_000_000) == pytest.approx(24.0)
    assert mc.cost_usd("claude-fable-5-1", 2_000, 1_000) == pytest.approx(0.07)


def test_tasks_mention_claude_code_rule():
    # CLAUDE.md: AI tooling references in cover letters must name Claude Code
    assert "Claude Code" in mc.TASKS["cover_letter"]["prompt"]


def test_result_metrics():
    r = mc.Result(model="claude-opus-5", task="job_fit", effort="high",
                  input_tokens=1000, output_tokens=500, latency_s=10.0)
    assert r.tokens_per_s == 50
    assert r.cost == pytest.approx((1000 * 5 + 500 * 25) / 1_000_000)
    assert mc.Result(model="claude-opus-5", task="job_fit", effort="high").tokens_per_s == 0


def test_report_includes_rows_totals_and_errors():
    results = [
        mc.Result("claude-opus-5-5", "job_fit", "medium", 800, 400, 5.0, 1.2, "end_turn", "Fit: 7/10"),
        mc.Result("claude-fable-5-1", "job_fit", "high", error="429: rate limited"),
    ]
    report = mc.render_report(results)
    assert "| job_fit | Claude Opus 5.5 | medium | 5.0 | 1.2 | 80 | 800 | 400 |" in report
    assert "error: 429: rate limited" in report
    assert "Fit: 7/10" in report
    # Errored model has no totals row
    totals = report.split("## Totals per model")[1].split("## Outputs")[0]
    assert "Claude Opus 5.5" in totals and "Claude Fable 5.1" not in totals


class _FakeUsage:
    input_tokens = 120
    output_tokens = 60


class _FakeBlock:
    def __init__(self, type_, text=""):
        self.type, self.text = type_, text


class _FakeMessage:
    usage = _FakeUsage()
    stop_reason = "end_turn"
    content = [_FakeBlock("thinking"), _FakeBlock("text", "Hello"), _FakeBlock("text", " world")]


class _FakeStream:
    text_stream = iter(["Hello", " world"])

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get_final_message(self):
        return _FakeMessage()


class _FakeClient:
    def __init__(self):
        self.calls = []
        self.messages = self

    def stream(self, **params):
        self.calls.append(params)
        return _FakeStream()


def test_run_one_uses_default_effort_and_no_fallbacks():
    pytest.importorskip("anthropic")
    client = _FakeClient()
    r = mc.run_one(client, "claude-opus-5-5", "job_fit", None, 16000)
    params = client.calls[0]
    assert params["model"] == "claude-opus-5-5"
    assert params["output_config"] == {"effort": "medium"}
    assert "fallbacks" not in params and "thinking" not in params
    assert r.text == "Hello world"
    assert (r.input_tokens, r.output_tokens, r.stop_reason) == (120, 60, "end_turn")
    assert r.ttft_s is not None and r.latency_s >= r.ttft_s


def test_run_one_effort_override():
    pytest.importorskip("anthropic")
    client = _FakeClient()
    r = mc.run_one(client, "claude-fable-5-1", "interview_prep", "low", 16000)
    assert client.calls[0]["output_config"] == {"effort": "low"}
    assert r.effort == "low"
