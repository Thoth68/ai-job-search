#!/usr/bin/env python3
"""
Claude Model Comparison Test

Runs the same job-application tasks against Claude Opus 5.5, Claude Opus 5
and Claude Fable 5.1 and compares latency, output speed, token usage and
cost, so you can decide which model to use for this workflow.

Based on The Rundown's guide "Claude Opus 5.5 vs Opus 5 vs Fable 5.1":
  - Opus 5.5: $4 / $20 per 1M tokens, default effort medium, >30% faster
    output than Opus 5, "Fable 5.1 level on most work"
  - Opus 5:   $5 / $25 per 1M tokens, default effort high
  - Fable 5.1: $10 / $50 per 1M tokens, default effort high, for the
    hardest reasoning tasks
All three have a 1M-token context window.

Requires an Anthropic credential (ANTHROPIC_API_KEY or `ant auth login`).
Every live run costs real money - use --dry-run to see the plan first.

Usage:
    python model_comparison.py --dry-run
    python model_comparison.py
    python model_comparison.py --tasks job_fit --models claude-opus-5-5 claude-fable-5-1
    python model_comparison.py --effort high --out reports/comparison.md
"""

import argparse
import sys
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path

# Pricing in USD per 1M tokens and default API effort, per the guide.
MODELS = {
    "claude-opus-5-5": {"name": "Claude Opus 5.5", "input": 4.00, "output": 20.00, "default_effort": "medium"},
    "claude-opus-5": {"name": "Claude Opus 5", "input": 5.00, "output": 25.00, "default_effort": "high"},
    "claude-fable-5-1": {"name": "Claude Fable 5.1", "input": 10.00, "output": 50.00, "default_effort": "high"},
}

SAMPLE_POSTING = """\
Data Scientist - Energy Trading (Copenhagen, hybrid)
We are a Nordic energy trading company building forecasting models for
wind and solar production and day-ahead power prices.
Requirements:
- MSc or PhD in a quantitative field
- 2+ years of Python (pandas, scikit-learn), SQL
- Experience with time-series forecasting
- Clear communication with traders and non-technical stakeholders
Nice to have:
- Cloud (Azure/Databricks), MLOps, experience with LLM tooling
- Danish language skills
"""

SAMPLE_PROFILE = """\
Candidate: MSc in Physics, 3 years as a data analyst at a utility company
(Python, pandas, SQL, Power BI), built a load-forecasting prototype with
gradient boosting, uses Claude Code for agentic coding, speaks English and
intermediate Danish. No production MLOps experience.
"""

TASKS = {
    "job_fit": {
        "title": "Job fit evaluation",
        "prompt": (
            "Evaluate this candidate's fit for the job posting. Score skills match, "
            "experience match and culture match from 1-10 each, list the top 3 gaps, "
            "and give a go / no-go recommendation.\n\n"
            f"<posting>\n{SAMPLE_POSTING}</posting>\n\n<profile>\n{SAMPLE_PROFILE}</profile>"
        ),
    },
    "cover_letter": {
        "title": "Cover letter opening",
        "prompt": (
            "Write the opening paragraph (max 120 words) of a cover letter for this "
            "candidate and posting. Be specific, avoid cliches, and mention Claude Code "
            "by name when referring to AI tooling.\n\n"
            f"<posting>\n{SAMPLE_POSTING}</posting>\n\n<profile>\n{SAMPLE_PROFILE}</profile>"
        ),
    },
    "interview_prep": {
        "title": "Interview prep (reasoning)",
        "prompt": (
            "The candidate lacks production MLOps experience. Predict the three hardest "
            "interview questions this gap will trigger, and for each draft a STAR-format "
            "answer using only facts from the profile. Flag anything that would require "
            "the candidate to overstate their experience.\n\n"
            f"<posting>\n{SAMPLE_POSTING}</posting>\n\n<profile>\n{SAMPLE_PROFILE}</profile>"
        ),
    },
}


@dataclass
class Result:
    model: str
    task: str
    effort: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    ttft_s: float | None = None
    stop_reason: str = ""
    text: str = ""
    error: str = ""

    @property
    def cost(self) -> float:
        return cost_usd(self.model, self.input_tokens, self.output_tokens)

    @property
    def tokens_per_s(self) -> float:
        return self.output_tokens / self.latency_s if self.latency_s else 0.0


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    p = MODELS[model]
    return (input_tokens * p["input"] + output_tokens * p["output"]) / 1_000_000


def price_ratio(model: str, baseline: str) -> float:
    """Fractional price difference of `model` vs `baseline` (-0.2 = 20% cheaper)."""
    return MODELS[model]["input"] / MODELS[baseline]["input"] - 1


def run_one(client, model: str, task: str, effort: str | None, max_tokens: int) -> Result:
    import anthropic

    eff = effort or MODELS[model]["default_effort"]
    result = Result(model=model, task=task, effort=eff)
    # Deliberately no server-side `fallbacks`: a fallback would silently answer
    # with a different model and invalidate the comparison. Refusals are reported.
    params = dict(
        model=model,
        max_tokens=max_tokens,
        output_config={"effort": eff},
        messages=[{"role": "user", "content": TASKS[task]["prompt"]}],
    )
    start = time.perf_counter()
    try:
        with client.messages.stream(**params) as stream:
            for _ in stream.text_stream:
                if result.ttft_s is None:
                    result.ttft_s = time.perf_counter() - start
            msg = stream.get_final_message()
    except anthropic.APIStatusError as e:
        result.error = f"{e.status_code}: {e.message}"
        return result
    except anthropic.APIConnectionError as e:
        result.error = f"connection error: {e}"
        return result
    result.latency_s = time.perf_counter() - start
    result.input_tokens = msg.usage.input_tokens
    result.output_tokens = msg.usage.output_tokens
    result.stop_reason = msg.stop_reason or ""
    result.text = "".join(b.text for b in msg.content if b.type == "text")
    return result


def render_report(results: list[Result]) -> str:
    lines = [f"# Claude model comparison - {date.today().isoformat()}", ""]
    lines += [
        "| Task | Model | Effort | Latency (s) | First text (s) | Output tok/s | In tok | Out tok | Cost (USD) | Stop |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        name = MODELS[r.model]["name"]
        if r.error:
            lines.append(f"| {r.task} | {name} | {r.effort} | error: {r.error} |||||||")
            continue
        ttft = f"{r.ttft_s:.1f}" if r.ttft_s is not None else "-"
        lines.append(
            f"| {r.task} | {name} | {r.effort} | {r.latency_s:.1f} | {ttft} | "
            f"{r.tokens_per_s:.0f} | {r.input_tokens} | {r.output_tokens} | "
            f"${r.cost:.4f} | {r.stop_reason} |"
        )

    lines += ["", "## Totals per model", "", "| Model | Total cost (USD) | Total latency (s) | Avg output tok/s |", "|---|---|---|---|"]
    for model in dict.fromkeys(r.model for r in results):
        ok = [r for r in results if r.model == model and not r.error]
        if not ok:
            continue
        total_cost = sum(r.cost for r in ok)
        total_lat = sum(r.latency_s for r in ok)
        avg_tps = sum(r.tokens_per_s for r in ok) / len(ok)
        lines.append(f"| {MODELS[model]['name']} | ${total_cost:.4f} | {total_lat:.1f} | {avg_tps:.0f} |")

    lines += ["", "## Outputs (judge quality yourself)", ""]
    for r in results:
        if r.error:
            continue
        lines += [f"### {TASKS[r.task]['title']} - {MODELS[r.model]['name']}", "", r.text.strip() or "_(no text)_", ""]
    return "\n".join(lines)


def print_plan(models: list[str], tasks: list[str], effort: str | None) -> None:
    print("Planned runs (no API calls made):")
    for t in tasks:
        for m in models:
            print(f"  {t:15} {MODELS[m]['name']:17} effort={effort or MODELS[m]['default_effort']}")
    print("\nPricing vs Claude Opus 5 (from the guide):")
    for m in models:
        print(f"  {MODELS[m]['name']:17} ${MODELS[m]['input']:.2f} in / ${MODELS[m]['output']:.2f} out "
              f"per 1M tokens ({price_ratio(m, 'claude-opus-5'):+.0%})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    ap.add_argument("--tasks", nargs="+", choices=list(TASKS), default=list(TASKS))
    ap.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"],
                    help="Force one effort level for all models (default: each model's API default)")
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--out", type=Path, help="Write the markdown report to this file")
    ap.add_argument("--dry-run", action="store_true", help="Show the plan and pricing without calling the API")
    args = ap.parse_args()

    if args.dry_run:
        print_plan(args.models, args.tasks, args.effort)
        return 0

    import anthropic

    client = anthropic.Anthropic()
    results = []
    for task in args.tasks:
        for model in args.models:
            print(f"Running {task} on {MODELS[model]['name']}...", file=sys.stderr)
            results.append(run_one(client, model, task, args.effort, args.max_tokens))

    report = render_report(results)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")
        print(f"Report written to {args.out}", file=sys.stderr)
    else:
        print(report)
    return 1 if all(r.error for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
