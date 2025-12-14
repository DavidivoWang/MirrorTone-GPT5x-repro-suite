import argparse
import csv
import datetime as dt
import json
import os
import random
import re
import sys
from pathlib import Path

import requests
import yaml

OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"


def read_cases(cases_path: Path):
    cases = []
    with cases_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cases.append(json.loads(line))
    return cases


def load_suite_spec(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def now_utc_compact():
    return dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def call_openai_responses(api_key: str, model: str, prompt: str, fixed_params: dict):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "input": [{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
        "temperature": fixed_params.get("temperature", 0.2),
        "top_p": fixed_params.get("top_p", 1.0),
        "max_output_tokens": fixed_params.get("max_output_tokens", 1200),
    }
    r = requests.post(OPENAI_RESPONSES_URL, headers=headers, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()

    text_parts = []
    for item in data.get("output", []):
        for c in item.get("content", []):
            if c.get("type") in ("output_text", "text"):
                text_parts.append(c.get("text", ""))
    text = "".join(text_parts).strip()
    usage = data.get("usage", {})
    return text, usage, data


def call_openai_chat(api_key: str, model: str, prompt: str, fixed_params: dict):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": fixed_params.get("temperature", 0.2),
        "top_p": fixed_params.get("top_p", 1.0),
        "max_tokens": fixed_params.get("max_output_tokens", 1200),
    }
    r = requests.post(OPENAI_CHAT_URL, headers=headers, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    text = (data.get("choices", [{}])[0].get("message", {}).get("content") or "").strip()
    usage = data.get("usage", {})
    return text, usage, data


def check_json_parse(output: str):
    try:
        json.loads(output)
        return True, ""
    except Exception as e:
        return False, f"json_parse_fail:{e.__class__.__name__}"


def check_no_extra_text(output: str):
    if "```" in output:
        return False, "contains_code_fence"
    return True, ""


def run_checks(case: dict, output: str, strict_no_extra_text: bool):
    checks = case.get("checks", [])
    passed_all = True
    notes = []
    hallucination_flags = 0

    if strict_no_extra_text:
        ok, msg = check_no_extra_text(output)
        if not ok:
            passed_all = False
            notes.append(msg)

    for chk in checks:
        ok, msg = True, ""
        if chk == "json_parse":
            ok, msg = check_json_parse(output)
        elif chk == "no_extra_text":
            ok, msg = check_no_extra_text(output)
        else:
            ok, msg = True, f"manual_check:{chk}"

        if not ok:
            passed_all = False
        if msg:
            notes.append(f"{chk}:{msg}")

    return passed_all, notes, hallucination_flags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--interface", choices=["responses", "chat"], required=True)
    ap.add_argument("--outdir", default="runs")
    ap.add_argument("--tag", default="run")
    args = ap.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Missing OPENAI_API_KEY", file=sys.stderr)
        sys.exit(2)

    root = Path(args.outdir)
    suite_spec = load_suite_spec(Path("suite/suite_spec.yaml"))
    cases = read_cases(Path("suite/cases.jsonl"))

    fixed_params = suite_spec.get("fixed_params", {})
    policy = suite_spec.get("run_policy", {})
    n_runs = int(policy.get("n_runs_per_case", 3))
    randomized = bool(policy.get("randomized_order", True))
    strict_no_extra_text = bool(suite_spec.get("scoring", {}).get("no_extra_text_strict", True))

    run_id = now_utc_compact()
    run_dir = root / run_id / args.tag / args.model
    ensure_dir(run_dir)

    manifest = {
        "run_id": run_id,
        "tag": args.tag,
        "model": args.model,
        "interface": args.interface,
        "fixed_params": fixed_params,
        "n_runs_per_case": n_runs,
        "randomized_order": randomized,
        "timestamp_utc": run_id,
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    case_list = cases[:]
    if randomized:
        random.shuffle(case_list)

    rows = []
    for case in case_list:
        case_id = case["case_id"]
        prompt = case["prompt"]

        for i in range(1, n_runs + 1):
            if args.interface == "responses":
                out_text, usage, raw = call_openai_responses(api_key, args.model, prompt, fixed_params)
            else:
                out_text, usage, raw = call_openai_chat(api_key, args.model, prompt, fixed_params)

            passed, notes, halluc_flags = run_checks(case, out_text, strict_no_extra_text)

            ensure_dir(run_dir / "cases")
            record = {
                "case_id": case_id,
                "run_index": i,
                "model": args.model,
                "interface": args.interface,
                "output_text": out_text,
                "usage": usage,
                "passed_all_checks": passed,
                "notes": notes,
                "hallucination_flags": halluc_flags,
            }
            (run_dir / "cases" / f"{case_id}.run{i}.json").write_text(
                json.dumps(record, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            rows.append({
                "run_id": run_id,
                "tag": args.tag,
                "model": args.model,
                "case_id": case_id,
                "task_success": 1 if passed else 0,
                "format_ok": 1 if passed else 0,
                "tokens_in": usage.get("input_tokens", usage.get("prompt_tokens", "")),
                "tokens_out": usage.get("output_tokens", usage.get("completion_tokens", "")),
                "hallucination_flags": halluc_flags,
                "notes": " | ".join(notes)[:1000],
            })

    results_path = run_dir / "results.csv"
    with results_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"Wrote: {results_path}")


if __name__ == "__main__":
    main()
