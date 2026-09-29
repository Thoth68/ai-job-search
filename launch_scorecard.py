#!/usr/bin/env python3
"""
Launch-Copy Scorecard: Claude Opus 5.5 vs Opus 5 vs Fable 5.1

Rebuilds the test from The Rundown's guide "Claude Opus 5.5 vs Opus 5 vs
Fable 5.1": each model rewrites an outdated launch draft from a fact sheet,
and the output is graded against the guide's checklist:

    Check                                   Opus 5.5   Opus 5   Fable 5.1
    Correct date, price, and trial terms    Pass/Fail  ...
    Both problems in the old draft caught   Pass/Fail  ...
    Product features described correctly    Pass/Fail  ...
    Length limits met                       Pass/Fail  ...
    Minutes spent editing                   Record     ...

The guide's own brief could not be retrieved, so the fact sheet and old
draft below are our own. The old draft has exactly two planted problems:
an outdated price ($9 beta price) and a feature the product does not have
(auto-applying to jobs). The first four checks are graded automatically;
"Minutes spent editing" is left for you to fill in after editing each draft.

Requires an Anthropic credential (ANTHROPIC_API_KEY or `ant auth login`).
Three API calls per run - use --dry-run to see the plan first.

Usage:
    python launch_scorecard.py --dry-run
    python launch_scorecard.py
    python launch_scorecard.py --out reports/launch_scorecard.md
"""

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from model_comparison import MODELS, cost_usd

FACT_SHEET = """\
Product: Hirely Pro (job-search assistant app)
Launch date: October 14, 2026
Price: $12 per month
Trial: 14-day free trial, no credit card required
Features (complete list - do not describe anything else):
  1. Tailors your CV to each job posting
  2. Application tracker with deadline reminders
  3. Mock interview practice with written feedback
"""

OLD_DRAFT = """\
Subject: Hirely Pro is here - land your next job faster

Hirely Pro launches October 14! For just $9 a month you get a smarter job
search: Hirely tailors your CV to every posting, keeps all your applications
in one tracker, and automatically applies to hundreds of jobs for you while
you sleep. Start your 14-day free trial today - no credit card required.
"""

LIMITS = {"subject": 60, "email_body_words": 150, "social_post": 280}

PROMPT = f"""\
We are launching Hirely Pro. Below is the approved fact sheet and an old draft
of the launch email written before the facts were finalised.

<fact_sheet>
{FACT_SHEET}</fact_sheet>

<old_draft>
{OLD_DRAFT}</old_draft>

1. Compare the old draft against the fact sheet and list every problem you find.
2. Write a corrected launch email: subject line of at most {LIMITS['subject']} characters,
   body of at most {LIMITS['email_body_words']} words.
3. Write a social media post of at most {LIMITS['social_post']} characters.
Use only facts from the fact sheet.
"""

SCHEMA = {
    "type": "object",
    "properties": {
        "problems_found": {"type": "array", "items": {"type": "string"}},
        "subject": {"type": "string"},
        "email_body": {"type": "string"},
        "social_post": {"type": "string"},
    },
    "required": ["problems_found", "subject", "email_body", "social_post"],
    "additionalProperties": False,
}

CHECKS = [
    "Correct date, price, and trial terms",
    "Both problems in the old draft caught",
    "Product features described correctly",
    "Length limits met",
]

# Grading patterns (case-insensitive)
DATE_RE = r"oct(ober|\.)?\s*14|14\s*oct(ober)?"
PRICE_RE = r"\$\s?12\b"
TRIAL_RE = r"14[\s-]*day"
NO_CARD_RE = r"no credit card"
WRONG_PRICE_RE = r"\$\s?9\b"
FAKE_FEATURE_RE = r"auto(matically)?[\s-]*appl|appl(y|ies|ying) (to|for) (hundreds of )?jobs for you|while you sleep"
REAL_FEATURE_RES = [r"\b(cv|resume|résumé)\b", r"track", r"interview"]
PROBLEM_PRICE_RE = r"\$\s?9|\$\s?12|price|pricing"
PROBLEM_FEATURE_RE = r"auto|appl(y|ies|ying)|feature|sleep"


@dataclass
class Score:
    model: str
    checks: dict = field(default_factory=dict)  # check -> (passed, note)
    output: dict = field(default_factory=dict)
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    error: str = ""

    @property
    def cost(self) -> float:
        return cost_usd(self.model, self.input_tokens, self.output_tokens)


def _has(pattern: str, text: str) -> bool:
    return re.search(pattern, text, re.IGNORECASE) is not None


def grade(output: dict) -> dict:
    """Grade one model's JSON output. Returns {check: (passed, note)}."""
    subject, body, social = output["subject"], output["email_body"], output["social_post"]
    copy = "\n".join([subject, body, social])
    results = {}

    missing = [name for name, pat in [("date", DATE_RE), ("price", PRICE_RE),
                                      ("14-day trial", TRIAL_RE), ("no credit card", NO_CARD_RE)]
               if not _has(pat, body)]
    wrong = ["$9 price"] if _has(WRONG_PRICE_RE, copy) else []
    results[CHECKS[0]] = (not missing and not wrong,
                          "; ".join(filter(None, [f"email missing: {', '.join(missing)}" if missing else "",
                                                  f"still has: {', '.join(wrong)}" if wrong else ""])) or "all correct")

    problems = " ".join(output["problems_found"])
    caught_price = _has(PROBLEM_PRICE_RE, problems)
    caught_feature = _has(PROBLEM_FEATURE_RE, problems)
    missed = [n for n, ok in [("wrong price", caught_price), ("invented auto-apply feature", caught_feature)] if not ok]
    results[CHECKS[1]] = (not missed, f"missed: {', '.join(missed)}" if missed else "both caught")

    fake = _has(FAKE_FEATURE_RE, copy)
    real = sum(_has(p, body) for p in REAL_FEATURE_RES)
    results[CHECKS[2]] = (not fake and real >= 2,
                          "; ".join(filter(None, ["mentions auto-apply" if fake else "",
                                                  f"only {real}/3 real features in email" if real < 2 else ""]))
                          or f"{real}/3 real features, nothing invented")

    words = len(body.split())
    over = []
    if len(subject) > LIMITS["subject"]:
        over.append(f"subject {len(subject)}>{LIMITS['subject']} chars")
    if words > LIMITS["email_body_words"]:
        over.append(f"body {words}>{LIMITS['email_body_words']} words")
    if len(social) > LIMITS["social_post"]:
        over.append(f"social {len(social)}>{LIMITS['social_post']} chars")
    results[CHECKS[3]] = (not over, "; ".join(over) or
                          f"subject {len(subject)}c, body {words}w, social {len(social)}c")
    return results


def run_one(client, model: str, effort: str | None, max_tokens: int) -> Score:
    import anthropic

    score = Score(model=model)
    # No server-side `fallbacks`: a fallback would answer with a different model
    # and invalidate the comparison. A refusal is reported as an error instead.
    start = time.perf_counter()
    try:
        with client.messages.stream(
            model=model,
            max_tokens=max_tokens,
            output_config={
                "effort": effort or MODELS[model]["default_effort"],
                "format": {"type": "json_schema", "schema": SCHEMA},
            },
            messages=[{"role": "user", "content": PROMPT}],
        ) as stream:
            msg = stream.get_final_message()
    except anthropic.APIStatusError as e:
        score.error = f"{e.status_code}: {e.message}"
        return score
    except anthropic.APIConnectionError as e:
        score.error = f"connection error: {e}"
        return score
    score.latency_s = time.perf_counter() - start
    score.input_tokens = msg.usage.input_tokens
    score.output_tokens = msg.usage.output_tokens
    if msg.stop_reason != "end_turn":
        score.error = f"stop_reason={msg.stop_reason}"
        return score
    text = "".join(b.text for b in msg.content if b.type == "text")
    try:
        score.output = json.loads(text)
    except json.JSONDecodeError as e:
        score.error = f"invalid JSON: {e}"
        return score
    score.checks = grade(score.output)
    return score


def render_report(scores: list[Score]) -> str:
    names = [MODELS[s.model]["name"] for s in scores]
    lines = [f"# Launch-copy scorecard - {date.today().isoformat()}", "",
             "| Check | " + " | ".join(names) + " |",
             "|---" * (len(scores) + 1) + "|"]
    for check in CHECKS:
        cells = []
        for s in scores:
            if s.error:
                cells.append("Fail (error)")
            else:
                cells.append("Pass" if s.checks[check][0] else "Fail")
        lines.append(f"| {check} | " + " | ".join(cells) + " |")
    lines.append("| Minutes spent editing | " + " | ".join("Record: ___" for _ in scores) + " |")
    lines.append("| Cost (USD) | " + " | ".join("-" if s.error else f"${s.cost:.4f}" for s in scores) + " |")
    lines.append("| Latency (s) | " + " | ".join("-" if s.error else f"{s.latency_s:.1f}" for s in scores) + " |")

    lines += ["", "## Grading notes", ""]
    for s in scores:
        lines.append(f"**{MODELS[s.model]['name']}**" + (f" - error: {s.error}" if s.error else ""))
        for check, (ok, note) in s.checks.items():
            lines.append(f"- {'Pass' if ok else 'Fail'} - {check}: {note}")
        lines.append("")

    lines += ["## Outputs (edit each, then record minutes spent)", ""]
    for s in scores:
        if s.error:
            continue
        o = s.output
        lines += [f"### {MODELS[s.model]['name']}", "", "Problems found:"]
        lines += [f"- {p}" for p in o["problems_found"]]
        lines += ["", f"**Subject:** {o['subject']}", "", o["email_body"].strip(), "",
                  f"**Social post:** {o['social_post']}", ""]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    ap.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"],
                    help="Force one effort level for all models (default: each model's API default)")
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--out", type=Path, help="Write the markdown report to this file")
    ap.add_argument("--dry-run", action="store_true", help="Show the prompt and plan without calling the API")
    args = ap.parse_args()

    if args.dry_run:
        print(PROMPT)
        for m in args.models:
            print(f"Would run: {MODELS[m]['name']} (effort={args.effort or MODELS[m]['default_effort']})")
        return 0

    import anthropic

    client = anthropic.Anthropic()
    scores = []
    for model in args.models:
        print(f"Running {MODELS[model]['name']}...", file=sys.stderr)
        scores.append(run_one(client, model, args.effort, args.max_tokens))

    report = render_report(scores)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")
        print(f"Report written to {args.out}", file=sys.stderr)
    else:
        print(report)
    return 1 if all(s.error for s in scores) else 0


if __name__ == "__main__":
    sys.exit(main())
