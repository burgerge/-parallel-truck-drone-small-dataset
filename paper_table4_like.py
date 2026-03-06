import argparse
import csv
from typing import Dict, List

from paper_eval_common import (
    build_cfg,
    ci95,
    clone_cfg,
    excluded_neighborhood,
    parse_instance_names,
    run_full_a1,
    save_json,
)


def run_table4_like(
    instances_root: str,
    instance_names: List[str],
    cfg_template,
) -> Dict:
    baseline = {}
    for name in instance_names:
        baseline[name] = run_full_a1(name, instances_root, clone_cfg(cfg_template), disable_improvement=False)

    out = {}
    for nid in range(1, 8):
        obj_inc = []
        cpu_dec = []
        rows = []
        with excluded_neighborhood(nid):
            for name in instance_names:
                res = run_full_a1(name, instances_root, clone_cfg(cfg_template), disable_improvement=False)
                base = baseline[name]

                di = (res["expected_obj"] - base["expected_obj"]) / base["expected_obj"] * 100.0
                dc = (base["runtime_sec"] - res["runtime_sec"]) / base["runtime_sec"] * 100.0
                obj_inc.append(di)
                cpu_dec.append(dc)
                rows.append(
                    {
                        "instance": name,
                        "excluded_neighborhood": nid,
                        "expected_obj": res["expected_obj"],
                        "runtime_sec": res["runtime_sec"],
                        "pct_obj_increase_vs_baseline": di,
                        "pct_cpu_decrease_vs_baseline": dc,
                    }
                )

        obj_stats = ci95(obj_inc)
        cpu_stats = ci95(cpu_dec)
        out[f"N{nid}"] = {
            "avg_pct_increase_objective": obj_stats["avg"],
            "ci95_objective": [obj_stats["low"], obj_stats["high"]],
            "avg_pct_decrease_cpu": cpu_stats["avg"],
            "ci95_cpu": [cpu_stats["low"], cpu_stats["high"]],
            "rows": rows,
        }
    return out


def save_table4_csv(path: str, table4: Dict) -> None:
    import os

    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "neighborhood",
                "avg_pct_increase_objective",
                "ci95_obj_low",
                "ci95_obj_high",
                "avg_pct_decrease_cpu",
                "ci95_cpu_low",
                "ci95_cpu_high",
            ]
        )
        for nid in range(1, 8):
            k = f"N{nid}"
            it = table4[k]
            w.writerow(
                [
                    k,
                    it["avg_pct_increase_objective"],
                    it["ci95_objective"][0],
                    it["ci95_objective"][1],
                    it["avg_pct_decrease_cpu"],
                    it["ci95_cpu"][0],
                    it["ci95_cpu"][1],
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Table4-like neighborhood exclusion experiment.")
    parser.add_argument("--instances-root", default="./Instance")
    parser.add_argument("--instances", default="1-10", help="Examples: 1-10 | 1,2,3 | Instance3,Instance4")
    parser.add_argument("--p", type=int, default=2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--k-max", type=int, default=7)
    parser.add_argument("--l-max", type=int, default=6)
    parser.add_argument("--i-max", type=int, default=5)
    parser.add_argument("--out-json", default="outputs/table4_like_current_code.json")
    parser.add_argument("--out-csv", default="outputs/table4_like_current_code.csv")
    args = parser.parse_args()

    instance_names = parse_instance_names(args.instances)
    cfg = build_cfg(args.p, args.seed, args.k_max, args.l_max, args.i_max)
    table4 = run_table4_like(args.instances_root, instance_names, cfg)
    payload = {
        "config": {
            "instances": instance_names,
            "p": args.p,
            "seed": args.seed,
            "k_max": args.k_max,
            "l_max": args.l_max,
            "i_max": args.i_max,
        },
        "table4_like": table4,
    }
    save_json(args.out_json, payload)
    save_table4_csv(args.out_csv, table4)

    print(f"Saved JSON: {args.out_json}")
    print(f"Saved CSV : {args.out_csv}")
    for nid in range(1, 8):
        it = table4[f"N{nid}"]
        print(
            f"N{nid}: obj_inc={it['avg_pct_increase_objective']:.4f}% "
            f"CI=({it['ci95_objective'][0]:.4f}, {it['ci95_objective'][1]:.4f}) | "
            f"cpu_dec={it['avg_pct_decrease_cpu']:.4f}% "
            f"CI=({it['ci95_cpu'][0]:.4f}, {it['ci95_cpu'][1]:.4f})"
        )


if __name__ == "__main__":
    main()
