import argparse
import json
import os
import statistics
import time
from typing import Dict, List, Optional, Sequence, Set, Tuple

from alg import (
    HeuristicConfig,
    algorithm_1_stochastic_lrp,
    algorithm_3_initial_construction,
    algorithm_5_vns_improvement,
    calculate_arrival_times,
)
from main import load_instance_data
from vns.types import Solution
from vns.validate import validate_solution


def _cfg_to_dict(cfg: HeuristicConfig) -> Dict:
    return {
        "num_depots_to_open": cfg.num_depots_to_open,
        "num_scenarios": cfg.num_scenarios,
        "seed": cfg.seed,
        "k_max": cfg.k_max,
        "l_max": cfg.l_max,
        "i_max": cfg.i_max,
        "drone_time_is_roundtrip": cfg.drone_time_is_roundtrip,
        "normalize_by_num_demands": cfg.normalize_by_num_demands,
        "infeasible_penalty": cfg.infeasible_penalty,
    }


def _parse_int_list(raw: Optional[str]) -> List[int]:
    if not raw:
        return []
    out = []
    for token in raw.replace(" ", "").split(","):
        if token == "":
            continue
        out.append(int(token))
    return out


def _parse_float_list(raw: Optional[str]) -> List[float]:
    if not raw:
        return []
    out = []
    for token in raw.replace(" ", "").split(","):
        if token == "":
            continue
        out.append(float(token))
    return out


def _normalize_weights(weights: Sequence[float], n: int) -> List[float]:
    if not weights:
        return [1.0 / n] * n
    if len(weights) != n:
        raise ValueError(f"weights length={len(weights)} but scenarios={n}")
    s = sum(weights)
    if s <= 0:
        raise ValueError("weights must sum to positive value")
    return [w / s for w in weights]


def _weighted_scenarios(scenarios: List, weights: List[float]) -> List[Tuple[object, float]]:
    return [(scenarios[i], weights[i]) for i in range(len(scenarios))]


def _default_x_open(candidates: List[int], p: int) -> Set[int]:
    if p <= 0:
        return set()
    return set(candidates[: min(p, len(candidates))])


def _map_x_to_base(x_open: Sequence[int], depot_base_map: Dict[int, int]) -> List[int]:
    return [int(depot_base_map.get(int(k), int(k))) for k in x_open]


def _scenario_eval(
    x_open: Set[int],
    scenario,
    cfg: HeuristicConfig,
) -> Dict:
    tt0, ds0 = algorithm_3_initial_construction(x_open, scenario)
    obj0 = calculate_arrival_times(tt0, ds0, scenario, cfg)
    tt1, ds1 = algorithm_5_vns_improvement(tt0, ds0, scenario, cfg)
    obj1 = calculate_arrival_times(tt1, ds1, scenario, cfg)
    vr = validate_solution(Solution(tt1, ds1), scenario)
    return {
        "obj_a3": obj0,
        "obj_a5": obj1,
        "improved": obj1 < obj0,
        "valid": vr.ok,
        "errors": vr.errors,
        "tt": tt1,
        "ds": ds1,
    }


def _run_fixed_x(
    instance_name: str,
    depot_base_map: Dict[int, int],
    scenarios: List,
    weights: List[float],
    x_open: Set[int],
    cfg: HeuristicConfig,
) -> Dict:
    rows = []
    expected_a3 = 0.0
    expected_a5 = 0.0
    for i, scen in enumerate(scenarios):
        r = _scenario_eval(x_open, scen, cfg)
        rows.append({"scenario_index": i, "weight": weights[i], **r})
        expected_a3 += weights[i] * r["obj_a3"]
        expected_a5 += weights[i] * r["obj_a5"]

    return {
        "mode": "fixed-x",
        "instance": instance_name,
        "config": _cfg_to_dict(cfg),
        "scenario_count": len(scenarios),
        "scenario_weights": list(weights),
        "x_open": sorted(x_open),
        "x_open_base": _map_x_to_base(sorted(x_open), depot_base_map),
        "depot_base_map": dict(depot_base_map),
        "expected_obj_a3": expected_a3,
        "expected_obj_a5": expected_a5,
        "improved": expected_a5 < expected_a3,
        "scenarios": rows,
    }


def _run_full_a1(
    base_instance,
    instance_name: str,
    scenarios: List,
    weights: List[float],
    p: int,
    seeds: List[int],
    cfg_template: HeuristicConfig,
    include_best_seed_details: bool,
) -> Dict:
    weighted = _weighted_scenarios(scenarios, weights)
    depot_base_map = dict(getattr(base_instance, "depot_base_map", {}))
    runs = []
    for seed in seeds:
        cfg = HeuristicConfig(
            num_depots_to_open=p,
            num_scenarios=len(scenarios),
            seed=seed,
            k_max=cfg_template.k_max,
            l_max=cfg_template.l_max,
            i_max=cfg_template.i_max,
            drone_time_is_roundtrip=cfg_template.drone_time_is_roundtrip,
            normalize_by_num_demands=cfg_template.normalize_by_num_demands,
            infeasible_penalty=cfg_template.infeasible_penalty,
        )
        t0 = time.perf_counter()
        best_x, best_e = algorithm_1_stochastic_lrp(base_instance, cfg=cfg, scenarios=weighted)
        elapsed = time.perf_counter() - t0
        runs.append(
            {
                "seed": seed,
                "best_x_open": sorted(best_x),
                "best_x_open_base": _map_x_to_base(sorted(best_x), depot_base_map),
                "expected_obj": best_e,
                "runtime_sec": elapsed,
            }
        )

    objs = [r["expected_obj"] for r in runs]
    best_idx = min(range(len(runs)), key=lambda i: runs[i]["expected_obj"])
    best_run = runs[best_idx]

    out = {
        "mode": "full-a1",
        "instance": instance_name,
        "config": _cfg_to_dict(cfg_template),
        "scenario_count": len(scenarios),
        "scenario_weights": list(weights),
        "p": p,
        "seeds": seeds,
        "runs": runs,
        "summary": {
            "best_expected_obj": min(objs),
            "avg_expected_obj": statistics.mean(objs),
            "std_expected_obj": statistics.pstdev(objs) if len(objs) > 1 else 0.0,
            "best_seed": best_run["seed"],
            "best_x_open": best_run["best_x_open"],
            "best_x_open_base": best_run["best_x_open_base"],
        },
        "depot_base_map": depot_base_map,
    }

    if include_best_seed_details:
        cfg = HeuristicConfig(
            num_depots_to_open=p,
            num_scenarios=len(scenarios),
            seed=best_run["seed"],
            k_max=cfg_template.k_max,
            l_max=cfg_template.l_max,
            i_max=cfg_template.i_max,
            drone_time_is_roundtrip=cfg_template.drone_time_is_roundtrip,
            normalize_by_num_demands=cfg_template.normalize_by_num_demands,
            infeasible_penalty=cfg_template.infeasible_penalty,
        )
        x_open = set(best_run["best_x_open"])
        detail_rows = []
        for i, scen in enumerate(scenarios):
            r = _scenario_eval(x_open, scen, cfg)
            detail_rows.append({"scenario_index": i, "weight": weights[i], **r})
        out["best_seed_scenario_details"] = detail_rows

    return out


def _print_fixed_x(result: Dict) -> None:
    print(
        f"\nMode: fixed-x | Instance: {result['instance']} | "
        f"X_open={result['x_open']} (base={result.get('x_open_base', result['x_open'])})"
    )
    print("scenario  w        obj_a3      obj_a5      improved  valid")
    for row in result["scenarios"]:
        print(
            f"{row['scenario_index']:>7}  "
            f"{row['weight']:<7.4f}  "
            f"{row['obj_a3']:<10.4f}  "
            f"{row['obj_a5']:<10.4f}  "
            f"{str(row['improved']):<8}  "
            f"{str(row['valid'])}"
        )
    print(
        f"Expected: A3={result['expected_obj_a3']:.4f} | "
        f"A5={result['expected_obj_a5']:.4f} | improved={result['improved']}"
    )


def _print_full_a1(result: Dict) -> None:
    print(f"\nMode: full-a1 | Instance: {result['instance']} | p={result['p']}")
    print("seed     expected_obj   runtime_sec   X_open(internal/base)")
    for row in result["runs"]:
        print(
            f"{row['seed']:<8} {row['expected_obj']:<13.4f} "
            f"{row['runtime_sec']:<11.3f} {row['best_x_open']}/{row.get('best_x_open_base', row['best_x_open'])}"
        )
    s = result["summary"]
    print(
        f"Summary: best={s['best_expected_obj']:.4f} avg={s['avg_expected_obj']:.4f} "
        f"std={s['std_expected_obj']:.4f} | best_seed={s['best_seed']} "
        f"| best_X_open={s['best_x_open']} (base={s.get('best_x_open_base', s['best_x_open'])})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run reproducible experiments for the parallel truck-drone heuristic."
    )
    parser.add_argument("--instances-root", default="./Instance")
    parser.add_argument("--instance", default="Instance3")
    parser.add_argument("--mode", choices=["fixed-x", "full-a1"], default="full-a1")
    parser.add_argument("--p", type=int, default=2, help="Number of depots to open.")
    parser.add_argument("--x-open", default="", help="Comma-separated depot ids for fixed-x mode.")
    parser.add_argument("--seeds", default="123", help="Comma-separated seeds for full-a1 mode.")
    parser.add_argument("--scenario-weights", default="", help="Comma-separated scenario weights.")
    parser.add_argument("--k-max", type=int, default=7)
    parser.add_argument("--l-max", type=int, default=6)
    parser.add_argument("--i-max", type=int, default=20)
    parser.add_argument("--normalize", choices=["avg", "sum"], default="avg")
    parser.add_argument("--drone-roundtrip", choices=["true", "false"], default="true")
    parser.add_argument("--out", default="", help="Optional output JSON path.")
    parser.add_argument(
        "--include-best-seed-details",
        action="store_true",
        help="In full-a1 mode, save scenario-level TTk/DSk for the best seed.",
    )
    args = parser.parse_args()

    folder = os.path.join(args.instances_root, args.instance)
    if not os.path.exists(folder):
        raise FileNotFoundError(f"Instance folder not found: {folder}")

    base_instance, scenarios = load_instance_data(folder)
    if not scenarios:
        raise ValueError("No scenario was loaded from t.txt.")

    weights = _normalize_weights(_parse_float_list(args.scenario_weights), len(scenarios))
    cfg_template = HeuristicConfig(
        num_depots_to_open=args.p,
        num_scenarios=len(scenarios),
        seed=None,
        k_max=args.k_max,
        l_max=args.l_max,
        i_max=args.i_max,
        drone_time_is_roundtrip=(args.drone_roundtrip.lower() == "true"),
        normalize_by_num_demands=(args.normalize == "avg"),
    )

    if args.mode == "fixed-x":
        x_open_raw = _parse_int_list(args.x_open)
        x_open = set(x_open_raw) if x_open_raw else _default_x_open(base_instance.candidate_depots, args.p)
        cfg = HeuristicConfig(
            num_depots_to_open=args.p,
            num_scenarios=len(scenarios),
            seed=None,
            k_max=cfg_template.k_max,
            l_max=cfg_template.l_max,
            i_max=cfg_template.i_max,
            drone_time_is_roundtrip=cfg_template.drone_time_is_roundtrip,
            normalize_by_num_demands=cfg_template.normalize_by_num_demands,
            infeasible_penalty=cfg_template.infeasible_penalty,
        )
        result = _run_fixed_x(
            args.instance,
            dict(getattr(base_instance, "depot_base_map", {})),
            scenarios,
            weights,
            x_open,
            cfg,
        )
        _print_fixed_x(result)
    else:
        seeds = _parse_int_list(args.seeds)
        if not seeds:
            seeds = [123]
        result = _run_full_a1(
            base_instance=base_instance,
            instance_name=args.instance,
            scenarios=scenarios,
            weights=weights,
            p=args.p,
            seeds=seeds,
            cfg_template=cfg_template,
            include_best_seed_details=args.include_best_seed_details,
        )
        _print_full_a1(result)

    if args.out:
        out_dir = os.path.dirname(args.out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"Saved report to: {args.out}")


if __name__ == "__main__":
    main()
