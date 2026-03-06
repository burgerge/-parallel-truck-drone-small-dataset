import argparse
import csv
import os
import statistics
import time
from contextlib import contextmanager
from itertools import combinations
from typing import Dict, List, Sequence, Set, Tuple

import alg
from main import load_instance_data
from paper_eval_common import build_cfg, parse_instance_names, save_json


@contextmanager
def no_vns_improvement():
    original = alg.algorithm_5_vns_improvement

    def _noop(tt_k, ds_k, instance, cfg):
        return tt_k, ds_k

    alg.algorithm_5_vns_improvement = _noop
    try:
        yield
    finally:
        alg.algorithm_5_vns_improvement = original


def _set_to_sorted_list(x: Set[int]) -> List[int]:
    return sorted(int(v) for v in x)


def _to_base_ids(x_open: Sequence[int], depot_base_map: Dict[int, int]) -> List[int]:
    return [int(depot_base_map.get(int(k), int(k))) for k in x_open]


def _to_candidate_indices(x_open: Sequence[int], candidate_order: Sequence[int]) -> List[int]:
    pos = {int(k): i for i, k in enumerate(candidate_order)}
    return sorted(pos[int(k)] for k in x_open)


def _algorithm1_once(
    base_instance,
    scenarios,
    cfg: alg.HeuristicConfig,
    disable_improvement: bool,
) -> Tuple[List[int], float, float]:
    t0 = time.perf_counter()
    if disable_improvement:
        with no_vns_improvement():
            best_x, best_obj = alg.algorithm_1_stochastic_lrp(base_instance, cfg=cfg, scenarios=scenarios)
    else:
        best_x, best_obj = alg.algorithm_1_stochastic_lrp(base_instance, cfg=cfg, scenarios=scenarios)
    elapsed = time.perf_counter() - t0
    return _set_to_sorted_list(best_x), float(best_obj), float(elapsed)


def _enumeration_benchmark(
    base_instance,
    scenarios,
    p: int,
    cfg: alg.HeuristicConfig,
    disable_improvement: bool,
) -> Tuple[List[int], float, float]:
    depots = list(base_instance.candidate_depots)
    best_x: List[int] = []
    best_obj = float("inf")
    t0 = time.perf_counter()

    for combo in combinations(depots, p):
        x_set = set(combo)
        if disable_improvement:
            with no_vns_improvement():
                f = alg.algorithm_2_depot_evaluation(x_set, base_instance, scenarios, cfg)
        else:
            f = alg.algorithm_2_depot_evaluation(x_set, base_instance, scenarios, cfg)
        if f < best_obj:
            best_obj = float(f)
            best_x = sorted(int(v) for v in combo)

    elapsed = time.perf_counter() - t0
    return best_x, best_obj, float(elapsed)


def run_table3_aligned(
    instances_root: str,
    instance_names: List[str],
    cfg_template: alg.HeuristicConfig,
    benchmark_disable_improvement: bool = False,
) -> Dict:
    rows = []

    for name in instance_names:
        folder = os.path.join(instances_root, name)
        if not os.path.exists(folder):
            raise FileNotFoundError(f"Instance folder not found: {folder}")

        base_instance, scenarios = load_instance_data(folder)
        if not scenarios:
            raise ValueError(f"No scenarios loaded for {name}")

        depot_base_map = dict(getattr(base_instance, "depot_base_map", {}))
        candidate_order = list(base_instance.candidate_depots)

        cfg = alg.HeuristicConfig(
            num_depots_to_open=cfg_template.num_depots_to_open,
            num_scenarios=cfg_template.num_scenarios,
            seed=cfg_template.seed,
            k_max=cfg_template.k_max,
            l_max=cfg_template.l_max,
            i_max=cfg_template.i_max,
            drone_time_is_roundtrip=cfg_template.drone_time_is_roundtrip,
            normalize_by_num_demands=cfg_template.normalize_by_num_demands,
            infeasible_penalty=cfg_template.infeasible_penalty,
        )

        bench_x, bench_obj, bench_cpu = _enumeration_benchmark(
            base_instance,
            scenarios,
            cfg.num_depots_to_open,
            cfg,
            disable_improvement=benchmark_disable_improvement,
        )
        with_x, with_obj, with_cpu = _algorithm1_once(base_instance, scenarios, cfg, disable_improvement=False)
        wo_x, wo_obj, wo_cpu = _algorithm1_once(base_instance, scenarios, cfg, disable_improvement=True)

        gap_with = (with_obj - bench_obj) / bench_obj * 100.0 if bench_obj > 0 else 0.0
        gap_wo = (wo_obj - bench_obj) / bench_obj * 100.0 if bench_obj > 0 else 0.0

        rows.append(
            {
                "instance": name,
                "benchmark": {
                    "open_depots": bench_x,
                    "open_depots_base": _to_base_ids(bench_x, depot_base_map),
                    "open_depots_idx": _to_candidate_indices(bench_x, candidate_order),
                    "expected_obj": bench_obj,
                    "runtime_sec": bench_cpu,
                    "disable_improvement": bool(benchmark_disable_improvement),
                },
                "with_improvement": {
                    "open_depots": with_x,
                    "open_depots_base": _to_base_ids(with_x, depot_base_map),
                    "open_depots_idx": _to_candidate_indices(with_x, candidate_order),
                    "expected_obj": with_obj,
                    "runtime_sec": with_cpu,
                    "pct_gap_vs_benchmark": gap_with,
                },
                "without_improvement": {
                    "open_depots": wo_x,
                    "open_depots_base": _to_base_ids(wo_x, depot_base_map),
                    "open_depots_idx": _to_candidate_indices(wo_x, candidate_order),
                    "expected_obj": wo_obj,
                    "runtime_sec": wo_cpu,
                    "pct_gap_vs_benchmark": gap_wo,
                },
                "depot_base_map": depot_base_map,
            }
        )

    avg = {
        "benchmark_expected_obj": statistics.mean(r["benchmark"]["expected_obj"] for r in rows),
        "benchmark_runtime_sec": statistics.mean(r["benchmark"]["runtime_sec"] for r in rows),
        "with_improvement_gap_pct": statistics.mean(r["with_improvement"]["pct_gap_vs_benchmark"] for r in rows),
        "with_improvement_runtime_sec": statistics.mean(r["with_improvement"]["runtime_sec"] for r in rows),
        "without_improvement_gap_pct": statistics.mean(r["without_improvement"]["pct_gap_vs_benchmark"] for r in rows),
        "without_improvement_runtime_sec": statistics.mean(r["without_improvement"]["runtime_sec"] for r in rows),
    }
    return {"rows": rows, "average": avg}


def save_table3_aligned_csv(path: str, table3: Dict) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "instance",
                "benchmark_open_depots",
                "benchmark_open_depots_base",
                "benchmark_open_depots_idx",
                "benchmark_expected_obj",
                "benchmark_runtime_sec",
                "with_open_depots",
                "with_open_depots_base",
                "with_open_depots_idx",
                "with_expected_obj",
                "with_gap_vs_benchmark_pct",
                "with_runtime_sec",
                "without_open_depots",
                "without_open_depots_base",
                "without_open_depots_idx",
                "without_expected_obj",
                "without_gap_vs_benchmark_pct",
                "without_runtime_sec",
            ]
        )
        for r in table3["rows"]:
            w.writerow(
                [
                    r["instance"],
                    r["benchmark"]["open_depots"],
                    r["benchmark"]["open_depots_base"],
                    r["benchmark"]["open_depots_idx"],
                    r["benchmark"]["expected_obj"],
                    r["benchmark"]["runtime_sec"],
                    r["with_improvement"]["open_depots"],
                    r["with_improvement"]["open_depots_base"],
                    r["with_improvement"]["open_depots_idx"],
                    r["with_improvement"]["expected_obj"],
                    r["with_improvement"]["pct_gap_vs_benchmark"],
                    r["with_improvement"]["runtime_sec"],
                    r["without_improvement"]["open_depots"],
                    r["without_improvement"]["open_depots_base"],
                    r["without_improvement"]["open_depots_idx"],
                    r["without_improvement"]["expected_obj"],
                    r["without_improvement"]["pct_gap_vs_benchmark"],
                    r["without_improvement"]["runtime_sec"],
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Table3 aligned to paper-style metric: GAP vs enumerated best depot set."
    )
    parser.add_argument("--instances-root", default="./Instance")
    parser.add_argument("--instances", default="1-10", help="Examples: 1-10 | 1,2,3 | Instance3,Instance4")
    parser.add_argument("--p", type=int, default=2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--k-max", type=int, default=7)
    parser.add_argument("--l-max", type=int, default=6)
    parser.add_argument("--i-max", type=int, default=5)
    parser.add_argument(
        "--benchmark-disable-improvement",
        action="store_true",
        help="If set, enumerated benchmark also disables Algorithm 5.",
    )
    parser.add_argument("--out-json", default="outputs/table3_aligned_to_paper_metric.json")
    parser.add_argument("--out-csv", default="outputs/table3_aligned_to_paper_metric.csv")
    args = parser.parse_args()

    instance_names = parse_instance_names(args.instances)
    cfg = build_cfg(args.p, args.seed, args.k_max, args.l_max, args.i_max)

    table3 = run_table3_aligned(
        instances_root=args.instances_root,
        instance_names=instance_names,
        cfg_template=cfg,
        benchmark_disable_improvement=args.benchmark_disable_improvement,
    )
    payload = {
        "config": {
            "instances": instance_names,
            "p": args.p,
            "seed": args.seed,
            "k_max": args.k_max,
            "l_max": args.l_max,
            "i_max": args.i_max,
            "benchmark_disable_improvement": bool(args.benchmark_disable_improvement),
        },
        "table3_aligned": table3,
    }

    save_json(args.out_json, payload)
    save_table3_aligned_csv(args.out_csv, table3)

    avg = table3["average"]
    print(f"Saved JSON: {args.out_json}")
    print(f"Saved CSV : {args.out_csv}")
    print(
        "Average -> benchmark_cpu={:.3f}s, with_gap={:.3f}%, with_cpu={:.3f}s, "
        "wo_gap={:.3f}%, wo_cpu={:.3f}s".format(
            avg["benchmark_runtime_sec"],
            avg["with_improvement_gap_pct"],
            avg["with_improvement_runtime_sec"],
            avg["without_improvement_gap_pct"],
            avg["without_improvement_runtime_sec"],
        )
    )


if __name__ == "__main__":
    main()
