import argparse
import csv
import statistics
from pathlib import Path


def find_latest_run_dir(runs_dir: Path):
    candidates = [p for p in runs_dir.iterdir() if p.is_dir()]
    if not candidates:
        raise RuntimeError("No runs found")
    return sorted(candidates)[-1]


def load_results_csv(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def aggregate(rows):
    by_case = {}
    for r in rows:
        cid = r["case_id"]
        by_case.setdefault(cid, []).append(r)

    agg = {}
    for cid, items in by_case.items():
        def fnum(k):
            vals = []
            for it in items:
                try:
                    vals.append(float(it[k]))
                except Exception:
                    pass
            return statistics.mean(vals) if vals else 0.0

        agg[cid] = {
            "task_success": fnum("task_success"),
            "format_ok": fnum("format_ok"),
            "hallucination_flags": fnum("hallucination_flags"),
        }
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--runs_dir", default="runs")
    ap.add_argument("--out_md", default="docs/delta_table.md")
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    latest = find_latest_run_dir(runs_dir)

    base_csv = latest / "baseline" / args.baseline / "results.csv"
    cand_csv = latest / "candidate" / args.candidate / "results.csv"

    if not base_csv.exists():
        raise RuntimeError(f"Missing baseline results: {base_csv}")
    if not cand_csv.exists():
        raise RuntimeError(f"Missing candidate results: {cand_csv}")

    base = aggregate(load_results_csv(base_csv))
    cand = aggregate(load_results_csv(cand_csv))

    case_ids = sorted(set(base.keys()) | set(cand.keys()))

    lines = []
    lines.append(f"# Delta Table — {args.baseline} vs {args.candidate}")
    lines.append("")
    lines.append(f"Run folder: `{latest.name}`")
    lines.append("")
    lines.append("| case_id | TASK_SUCCESS (base) | TASK_SUCCESS (cand) | Δ | FORMAT_OK (base) | FORMAT_OK (cand) | Δ | HALLUC_FLAGS (base) | HALLUC_FLAGS (cand) | Δ |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    def get(m, cid, k):
        return m.get(cid, {}).get(k, 0.0)

    for cid in case_ids:
        b_ts = get(base, cid, "task_success")
        c_ts = get(cand, cid, "task_success")
        b_fm = get(base, cid, "format_ok")
        c_fm = get(cand, cid, "format_ok")
        b_hl = get(base, cid, "hallucination_flags")
        c_hl = get(cand, cid, "hallucination_flags")

        lines.append(
            f"| {cid} | {b_ts:.2f} | {c_ts:.2f} | {(c_ts-b_ts):+.2f} | "
            f"{b_fm:.2f} | {c_fm:.2f} | {(c_fm-b_fm):+.2f} | "
            f"{b_hl:.2f} | {c_hl:.2f} | {(c_hl-b_hl):+.2f} |"
        )

    out = Path(args.out_md)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
