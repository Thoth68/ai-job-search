"""Offline tests for launch_scorecard.py - no API calls, no cost.

Run with: python -m pytest tests/test_launch_scorecard.py
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import launch_scorecard as ls  # noqa: E402

GOOD = {
    "problems_found": [
        "Price is wrong: the draft says $9 a month, the fact sheet says $12.",
        "The draft claims Hirely automatically applies to jobs for you - that feature does not exist.",
    ],
    "subject": "Hirely Pro launches October 14",
    "email_body": (
        "Hirely Pro launches on October 14. It tailors your CV to each job posting, "
        "tracks every application with deadline reminders, and lets you practise mock "
        "interviews with written feedback. It costs $12 per month, and you can start "
        "with a 14-day free trial - no credit card required."
    ),
    "social_post": "Hirely Pro is live Oct 14: tailored CVs, an application tracker and mock interviews. $12/month, 14-day free trial, no credit card required.",
}


def _passed(output):
    return {check: ok for check, (ok, _) in ls.grade(output).items()}


def test_good_rewrite_passes_every_check():
    assert all(_passed(GOOD).values())


def test_old_draft_fails_facts_and_features():
    subject, body = ls.OLD_DRAFT.split("\n", 1)
    old = {"problems_found": [], "subject": subject.removeprefix("Subject: "),
           "email_body": body, "social_post": ""}
    passed = _passed(old)
    assert not passed["Correct date, price, and trial terms"]  # $9
    assert not passed["Both problems in the old draft caught"]
    assert not passed["Product features described correctly"]  # auto-apply
    assert passed["Length limits met"]


def test_old_draft_has_exactly_the_two_planted_problems():
    assert "$9" in ls.OLD_DRAFT and "automatically applies" in ls.OLD_DRAFT
    assert "October 14" in ls.OLD_DRAFT and "14-day free trial" in ls.OLD_DRAFT
    assert "no credit card required" in ls.OLD_DRAFT


def test_missing_trial_terms_fails_facts():
    out = dict(GOOD, email_body=GOOD["email_body"].replace(", and you can start with a 14-day free trial - no credit card required", ""))
    ok, note = ls.grade(out)["Correct date, price, and trial terms"]
    assert not ok and "14-day trial" in note and "no credit card" in note


def test_only_one_problem_caught_fails():
    out = dict(GOOD, problems_found=[GOOD["problems_found"][0]])
    ok, note = ls.grade(out)["Both problems in the old draft caught"]
    assert not ok and "auto-apply" in note


def test_invented_feature_fails():
    out = dict(GOOD, social_post="Hirely Pro auto-applies to jobs for you! $12/month.")
    assert not _passed(out)["Product features described correctly"]


def test_length_limits():
    out = dict(GOOD, subject="x" * 61, email_body="word " * 151, social_post="y" * 281)
    ok, note = ls.grade(out)["Length limits met"]
    assert not ok and "subject 61>60" in note and "body 151>150" in note and "social 281>280" in note


def test_report_matches_guide_table():
    good = ls.Score("claude-opus-5-5", checks=ls.grade(GOOD), output=GOOD,
                    input_tokens=500, output_tokens=300, latency_s=4.0)
    bad = ls.Score("claude-fable-5-1", error="stop_reason=refusal")
    report = ls.render_report([good, bad])
    assert "| Check | Claude Opus 5.5 | Claude Fable 5.1 |" in report
    assert "| Correct date, price, and trial terms | Pass | Fail (error) |" in report
    assert "| Minutes spent editing | Record: ___ | Record: ___ |" in report
    assert "error: stop_reason=refusal" in report


class _Usage:
    input_tokens, output_tokens = 400, 250


class _Block:
    def __init__(self, type_, text=""):
        self.type, self.text = type_, text


class _Stream:
    def __init__(self, msg):
        self.msg = msg

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get_final_message(self):
        return self.msg


class _Client:
    def __init__(self, stop_reason="end_turn", text=json.dumps(GOOD)):
        self.calls = []
        self.messages = self
        self.msg = type("M", (), {"usage": _Usage(), "stop_reason": stop_reason,
                                  "content": [_Block("thinking"), _Block("text", text)]})()

    def stream(self, **params):
        self.calls.append(params)
        return _Stream(self.msg)


def test_run_one_requests_structured_output_and_grades():
    pytest.importorskip("anthropic")
    client = _Client()
    s = ls.run_one(client, "claude-opus-5", None, 16000)
    cfg = client.calls[0]["output_config"]
    assert cfg["effort"] == "high" and cfg["format"]["type"] == "json_schema"
    assert "fallbacks" not in client.calls[0]
    assert not s.error and all(ok for ok, _ in s.checks.values())


def test_run_one_reports_refusal():
    pytest.importorskip("anthropic")
    s = ls.run_one(_Client(stop_reason="refusal", text=""), "claude-fable-5-1", None, 16000)
    assert s.error == "stop_reason=refusal" and not s.checks
