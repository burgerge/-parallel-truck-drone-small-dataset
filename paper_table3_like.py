import argparse
import csv
import statistics
from typing import Dict, List

from paper_eval_common import build_cfg, clone_cfg, parse_instance_names, run_full_a1, save_json


def run_table3_like(
    instances_root: str,
    instance_names: List[str],
    cfg_template,
) -> Dict:
    rows = []
    for name in instance_names:
        with_res = run_full_a1(name, instances_root, clone_cfg(cfg_template), disable_improvement=False)
        wo_res = run_full_a1(name, instances_root, clone_cfg(cfg_template), disable_improvement=True)

        gap = (wo_res["expected_obj"] - with_res["expected_obj"]) / with_res["expected_obj"] * 100.0
        cpu_dec = (with_res["runtime_sec"] - wo_res["runtime_sec"]) / with_res["runtime_sec"] * 100.0

        rows.append(
            {
                "instance": name,
                "with_improvement": with_res,
                "without_improvement": wo_res,
                "pct_gap_wo_vs_with": gap,
                "pct_cpu_decrease_wo_vs_with": cpu_dec,
            }
        )

    avg = {
        "with_improvement_expected_obj": statistics.mean(r["with_improvement"]["expected_obj"] for r in rows),
        "with_improvement_runtime_sec": statistics.mean(r["with_improvement"]["runtime_sec"] for r in rows),
        "without_improvement_expected_obj": statistics.mean(r["without_improvement"]["expected_obj"] for r in rows),
        "without_improvement_runtime_sec": statistics.mean(r["without_improvement"]["runtime_sec"] for r in rows),
        "avg_pct_gap_wo_vs_with": statistics.mean(r["pct_gap_wo_vs_with"] for r in rows),
        "avg_pct_cpu_decrease_wo_vs_with": statistics.mean(r["pct_cpu_decrease_wo_vs_with"] for r in rows),
    }
    return {"rows": rows, "average": avg}


def save_table3_csv(path: str, table3: Dict) -> None:
    import os

    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "instance",
                "with_open_depots",
                "with_open_depots_base",
                "with_expected_obj",
                "with_runtime_sec",
                "without_open_depots",
                "without_open_depots_base",
                "without_expected_obj",
                "without_runtime_sec",
                "pct_gap_wo_vs_with",
                "pct_cpu_decrease_wo_vs_with",
            ]
        )
        for r in table3["rows"]:
            w.writerow(
                [
                    r["instance"],
                    r["with_improvement"]["best_x_open"],
                    r["with_improvement"]["best_x_open_base"],
                    r["with_improvement"]["expected_obj"],
                    r["with_improvement"]["runtime_sec"],
                    r["without_improvement"]["best_x_open"],
                    r["without_improvement"]["best_x_open_base"],
                    r["without_improvement"]["expected_obj"],
                    r["without_improvement"]["runtime_sec"],
                    r["pct_gap_wo_vs_with"],
                    r["pct_cpu_decrease_wo_vs_with"],
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Table3-like experiment with/without VNS improvement.")
    parser.add_argument("--instances-root", default="./Instance")
    parser.add_argument("--instances", default="1-10", help="Examples: 1-10 | 1,2,3 | Instance3,Instance4")
    parser.add_argument("--p", type=int, default=2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--k-max", type=int, default=7)
    parser.add_argument("--l-max", type=int, default=6)
    parser.add_argument("--i-max", type=int, default=5)
    parser.add_argument("--out-json", default="outputs/table3_like_current_code.json")
    parser.add_argument("--out-csv", default="outputs/table3_like_current_code.csv")
    args = parser.parse_args()

    instance_names = parse_instance_names(args.instances)
    cfg = build_cfg(args.p, args.seed, args.k_max, args.l_max, args.i_max)
    table3 = run_table3_like(args.instances_root, instance_names, cfg)
    payload = {
        "config": {
            "instances": instance_names,
            "p": args.p,
            "seed": args.seed,
            "k_max": args.k_max,
            "l_max": args.l_max,
            "i_max": args.i_max,
        },
        "table3_like": table3,
    }
    save_json(args.out_json, payload)
    save_table3_csv(args.out_csv, table3)

    avg = table3["average"]
    print(f"Saved JSON: {args.out_json}")
    print(f"Saved CSV : {args.out_csv}")
    print(
        "Average -> with_obj={:.4f}, with_cpu={:.3f}s, wo_obj={:.4f}, wo_cpu={:.3f}s, gap={:.2f}%, cpu_dec={:.2f}%".format(
            avg["with_improvement_expected_obj"],
            avg["with_improvement_runtime_sec"],
            avg["without_improvement_expected_obj"],
            avg["without_improvement_runtime_sec"],
            avg["avg_pct_gap_wo_vs_with"],
            avg["avg_pct_cpu_decrease_wo_vs_with"],
        )
    )


if __name__ == "__main__":
    main()
