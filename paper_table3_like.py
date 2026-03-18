import argparse
import csv
import itertools
import math
import random
import statistics
from typing import Dict, List, Optional, Sequence, Tuple

from paper_eval_common import build_cfg, clone_cfg, parse_instance_names, run_full_a1, save_json
from gurobi_exact_small_enumeration import _prepare_data, solve_fixed_combo_commodity


def parse_seeds(raw: str, fallback_seed: int) -> List[int]:
    raw = (raw or "").strip()
    if not raw:
        return [int(fallback_seed)]

    out: List[int] = []
    for token in raw.split(","):
        t = token.strip()
        if not t:
            continue
        if "-" in t:
            a, b = t.split("-", 1)
            start = int(a.strip())
            end = int(b.strip())
            step = 1 if start <= end else -1
            out.extend(list(range(start, end + step, step)))
        else:
            out.append(int(t))

    unique: List[int] = []
    seen = set()
    for s in out:
        if s in seen:
            continue
        seen.add(s)
        unique.append(s)
    return unique or [int(fallback_seed)]


def _is_paper_feasible(status_name: Optional[str], objective: Optional[float]) -> bool:
    if objective is None:
        return False
    name = (status_name or "").upper()
    # Treat solver runs with incumbent objective as usable final scores.
    return name in {"OPTIMAL", "SUBOPTIMAL", "TIME_LIMIT", "USER_OBJ_LIMIT", "INTERRUPTED"}


def _final_obj(res: Dict, final_metric: str) -> Optional[float]:
    if final_metric == "paper_exact":
        paper = res.get("paper_exact", {})
        if _is_paper_feasible(paper.get("status_name"), paper.get("objective")):
            return float(paper["objective"])
        return None
    return float(res.get("arrival_sim_obj", res.get("expected_obj")))


def _safe_pct_gap(with_val: Optional[float], wo_val: Optional[float]) -> Optional[float]:
    if with_val is None or wo_val is None:
        return None
    if with_val <= 0:
        return None
    return (wo_val - with_val) / with_val * 100.0


def _safe_pct_cpu_dec(with_cpu: float, wo_cpu: float) -> Optional[float]:
    if with_cpu <= 0:
        return None
    return (with_cpu - wo_cpu) / with_cpu * 100.0


def _mean_optional(values: Sequence[Optional[float]]) -> Optional[float]:
    valid = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not valid:
        return None
    return statistics.mean(valid)


def _paper_exact_for_combo(
    instance_name: str,
    instances_root: str,
    open_depots: Sequence[int],
    time_limit: float,
    mip_gap: float,
    threads: int,
    mip_focus: int,
    presolve: int,
    cuts: int,
    symmetry: int,
    data_cache: Dict[str, Dict],
    score_cache: Dict[Tuple[str, Tuple[int, ...]], Dict],
) -> Dict:
    combo = tuple(sorted(int(x) for x in open_depots))
    cache_key = (instance_name, combo)
    if cache_key in score_cache:
        return dict(score_cache[cache_key])

    if instance_name not in data_cache:
        base, scenarios, demand_nodes, candidate_depots = _prepare_data(
            instance_name=instance_name,
            instances_root=instances_root,
            max_scenarios=0,
            max_demands=0,
        )
        pos = {int(k): idx for idx, k in enumerate(candidate_depots)}
        data_cache[instance_name] = {
            "base": base,
            "scenarios": scenarios,
            "demand_nodes": demand_nodes,
            "candidate_depots": candidate_depots,
            "candidate_pos": pos,
            "depot_base_map": dict(getattr(base, "depot_base_map", {})),
        }

    data = data_cache[instance_name]
    solved = solve_fixed_combo_commodity(
        instance_name=instance_name,
        scenarios=data["scenarios"],
        demand_nodes=data["demand_nodes"],
        open_depots=list(combo),
        time_limit=float(time_limit),
        mip_gap=float(mip_gap),
        mip_gap_abs=None,
        threads=int(threads),
        output_flag=0,
        mip_focus=int(mip_focus),
        heuristics=-1.0,
        presolve=int(presolve),
        cuts=int(cuts),
        symmetry=int(symmetry),
        return_solution_details=False,
        fixed_binary_values=None,
    )

    depot_base_map = data["depot_base_map"]
    pos = data["candidate_pos"]
    out = {
        "status": solved.get("status"),
        "status_name": solved.get("status_name"),
        "objective": solved.get("objective"),
        "best_bound": solved.get("best_bound"),
        "runtime_sec": solved.get("runtime_sec"),
        "achieved_mip_gap_pct": solved.get("achieved_mip_gap_pct"),
        "open_depots_internal": list(combo),
        "open_depots_idx": sorted(pos[int(k)] for k in combo if int(k) in pos),
        "open_depots_base": [depot_base_map.get(int(k), int(k)) for k in combo],
        "is_feasible": _is_paper_feasible(solved.get("status_name"), solved.get("objective")),
    }
    score_cache[cache_key] = dict(out)
    return out


def _attach_paper_exact(
    run_res: Dict,
    *,
    instance_name: str,
    instances_root: str,
    time_limit: float,
    mip_gap: float,
    threads: int,
    mip_focus: int,
    presolve: int,
    cuts: int,
    symmetry: int,
    data_cache: Dict[str, Dict],
    score_cache: Dict[Tuple[str, Tuple[int, ...]], Dict],
) -> Dict:
    out = dict(run_res)
    out["arrival_sim_obj"] = float(run_res.get("expected_obj"))
    out["paper_exact"] = _paper_exact_for_combo(
        instance_name=instance_name,
        instances_root=instances_root,
        open_depots=run_res.get("best_x_open", []),
        time_limit=time_limit,
        mip_gap=mip_gap,
        threads=threads,
        mip_focus=mip_focus,
        presolve=presolve,
        cuts=cuts,
        symmetry=symmetry,
        data_cache=data_cache,
        score_cache=score_cache,
    )
    return out


def _repair_infeasible_paper_combo(
    run_res: Dict,
    *,
    instance_name: str,
    instances_root: str,
    time_limit: float,
    mip_gap: float,
    threads: int,
    mip_focus: int,
    presolve: int,
    cuts: int,
    symmetry: int,
    data_cache: Dict[str, Dict],
    score_cache: Dict[Tuple[str, Tuple[int, ...]], Dict],
) -> Dict:
    paper = run_res.get("paper_exact", {})
    if bool(paper.get("is_feasible")):
        return run_res

    if instance_name not in data_cache:
        _paper_exact_for_combo(
            instance_name=instance_name,
            instances_root=instances_root,
            open_depots=run_res.get("best_x_open", []),
            time_limit=time_limit,
            mip_gap=mip_gap,
            threads=threads,
            mip_focus=mip_focus,
            presolve=presolve,
            cuts=cuts,
            symmetry=symmetry,
            data_cache=data_cache,
            score_cache=score_cache,
        )
    data = data_cache[instance_name]
    candidates = [int(x) for x in data.get("candidate_depots", [])]
    p = len(run_res.get("best_x_open", []))
    if p <= 0 or p > len(candidates):
        return run_res

    best: Optional[Dict] = None
    for combo in itertools.combinations(candidates, p):
        scored = _paper_exact_for_combo(
            instance_name=instance_name,
            instances_root=instances_root,
            open_depots=list(combo),
            time_limit=time_limit,
            mip_gap=mip_gap,
            threads=threads,
            mip_focus=mip_focus,
            presolve=presolve,
            cuts=cuts,
            symmetry=symmetry,
            data_cache=data_cache,
            score_cache=score_cache,
        )
        if not bool(scored.get("is_feasible")):
            continue
        if best is None or float(scored["objective"]) < float(best["objective"]):
            best = scored

    if best is None:
        return run_res

    out = dict(run_res)
    out["paper_exact"] = dict(best)
    out["best_x_open"] = list(best.get("open_depots_internal", run_res.get("best_x_open", [])))
    out["best_x_open_idx"] = list(best.get("open_depots_idx", run_res.get("best_x_open_idx", [])))
    out["best_x_open_base"] = list(best.get("open_depots_base", run_res.get("best_x_open_base", [])))
    out["paper_exact_repaired_from_infeasible"] = True
    return out


def run_table3_like(
    instances_root: str,
    instance_names: List[str],
    cfg_template,
    final_metric: str = "paper_exact",
    paper_time_limit: float = 300.0,
    paper_mip_gap: float = 0.0,
    paper_threads: int = 0,
    paper_mip_focus: int = 1,
    paper_presolve: int = 2,
    paper_cuts: int = 2,
    paper_symmetry: int = 2,
    data_cache: Optional[Dict[str, Dict]] = None,
    score_cache: Optional[Dict[Tuple[str, Tuple[int, ...]], Dict]] = None,
) -> Dict:
    if data_cache is None:
        data_cache = {}
    if score_cache is None:
        score_cache = {}
    rows = []
    for name in instance_names:
        with_res = run_full_a1(name, instances_root, clone_cfg(cfg_template), disable_improvement=False)
        wo_res = run_full_a1(name, instances_root, clone_cfg(cfg_template), disable_improvement=True)
        with_res = _attach_paper_exact(
            with_res,
            instance_name=name,
            instances_root=instances_root,
            time_limit=paper_time_limit,
            mip_gap=paper_mip_gap,
            threads=paper_threads,
            mip_focus=paper_mip_focus,
            presolve=paper_presolve,
            cuts=paper_cuts,
            symmetry=paper_symmetry,
            data_cache=data_cache,
            score_cache=score_cache,
        )
        wo_res = _attach_paper_exact(
            wo_res,
            instance_name=name,
            instances_root=instances_root,
            time_limit=paper_time_limit,
            mip_gap=paper_mip_gap,
            threads=paper_threads,
            mip_focus=paper_mip_focus,
            presolve=paper_presolve,
            cuts=paper_cuts,
            symmetry=paper_symmetry,
            data_cache=data_cache,
            score_cache=score_cache,
        )
        with_res = _repair_infeasible_paper_combo(
            with_res,
            instance_name=name,
            instances_root=instances_root,
            time_limit=paper_time_limit,
            mip_gap=paper_mip_gap,
            threads=paper_threads,
            mip_focus=paper_mip_focus,
            presolve=paper_presolve,
            cuts=paper_cuts,
            symmetry=paper_symmetry,
            data_cache=data_cache,
            score_cache=score_cache,
        )
        wo_res = _repair_infeasible_paper_combo(
            wo_res,
            instance_name=name,
            instances_root=instances_root,
            time_limit=paper_time_limit,
            mip_gap=paper_mip_gap,
            threads=paper_threads,
            mip_focus=paper_mip_focus,
            presolve=paper_presolve,
            cuts=paper_cuts,
            symmetry=paper_symmetry,
            data_cache=data_cache,
            score_cache=score_cache,
        )

        with_final = _final_obj(with_res, final_metric)
        wo_final = _final_obj(wo_res, final_metric)
        if with_final is None and wo_final is None:
            selected_variant = None
            selected_open = None
            selected_obj = None
        elif wo_final is None or (with_final is not None and with_final <= wo_final):
            selected_variant = "with_improvement"
            selected_open = list(with_res.get("best_x_open", []))
            selected_obj = with_final
        else:
            selected_variant = "without_improvement"
            selected_open = list(wo_res.get("best_x_open", []))
            selected_obj = wo_final
        gap = _safe_pct_gap(with_final, wo_final)
        cpu_dec = _safe_pct_cpu_dec(float(with_res["runtime_sec"]), float(wo_res["runtime_sec"]))

        rows.append(
            {
                "instance": name,
                "with_improvement": with_res,
                "without_improvement": wo_res,
                "with_final_obj": with_final,
                "without_final_obj": wo_final,
                "final_metric": final_metric,
                "selected_variant_by_final_metric": selected_variant,
                "selected_open_depots_internal_by_final_metric": selected_open,
                "selected_final_obj": selected_obj,
                "pct_gap_wo_vs_with": gap,
                "pct_cpu_decrease_wo_vs_with": cpu_dec,
            }
        )

    avg = {
        "with_improvement_arrival_sim_obj": statistics.mean(
            float(r["with_improvement"]["arrival_sim_obj"]) for r in rows
        ),
        "with_improvement_final_obj": _mean_optional([r.get("with_final_obj") for r in rows]),
        "with_improvement_runtime_sec": statistics.mean(float(r["with_improvement"]["runtime_sec"]) for r in rows),
        "without_improvement_arrival_sim_obj": statistics.mean(
            float(r["without_improvement"]["arrival_sim_obj"]) for r in rows
        ),
        "without_improvement_final_obj": _mean_optional([r.get("without_final_obj") for r in rows]),
        "without_improvement_runtime_sec": statistics.mean(float(r["without_improvement"]["runtime_sec"]) for r in rows),
        "avg_pct_gap_wo_vs_with": _mean_optional([r.get("pct_gap_wo_vs_with") for r in rows]),
        "avg_pct_cpu_decrease_wo_vs_with": _mean_optional([r.get("pct_cpu_decrease_wo_vs_with") for r in rows]),
        "final_metric": final_metric,
    }
    return {"rows": rows, "average": avg}


def _table3_csv_header(include_seed: bool = False) -> List[str]:
    head = []
    if include_seed:
        head.append("seed")
    head.extend(
        [
            "instance",
            "with_open_depots_internal",
            "with_final_obj_col3",
            "with_open_depots",
            "with_open_depots_base",
            "with_final_obj",
            "with_arrival_sim_obj_search",
            "with_paper_exact_obj",
            "with_paper_exact_status",
            "with_paper_exact_feasible",
            "with_runtime_sec",
            "without_open_depots_internal",
            "without_open_depots",
            "without_open_depots_base",
            "without_final_obj",
            "without_arrival_sim_obj_search",
            "without_paper_exact_obj",
            "without_paper_exact_status",
            "without_paper_exact_feasible",
            "without_runtime_sec",
            "final_metric",
            "selected_variant_by_final_metric",
            "selected_open_depots_internal_by_final_metric",
            "selected_final_obj",
            "pct_gap_wo_vs_with",
            "pct_cpu_decrease_wo_vs_with",
        ]
    )
    return head


def _table3_csv_row(r: Dict, seed: int = None, include_seed: bool = False) -> List:
    row = []
    if include_seed:
        row.append(seed)
    with_res = r["with_improvement"]
    wo_res = r["without_improvement"]
    with_paper = with_res.get("paper_exact", {})
    wo_paper = wo_res.get("paper_exact", {})
    row.extend(
        [
            r["instance"],
            with_res["best_x_open"],
            r.get("with_final_obj"),
            with_res["best_x_open_idx"],
            with_res["best_x_open_base"],
            r.get("with_final_obj"),
            with_res.get("arrival_sim_obj"),
            with_paper.get("objective"),
            with_paper.get("status_name"),
            with_paper.get("is_feasible"),
            with_res["runtime_sec"],
            wo_res["best_x_open"],
            wo_res["best_x_open_idx"],
            wo_res["best_x_open_base"],
            r.get("without_final_obj"),
            wo_res.get("arrival_sim_obj"),
            wo_paper.get("objective"),
            wo_paper.get("status_name"),
            wo_paper.get("is_feasible"),
            wo_res["runtime_sec"],
            r.get("final_metric"),
            r.get("selected_variant_by_final_metric"),
            r.get("selected_open_depots_internal_by_final_metric"),
            r.get("selected_final_obj"),
            r["pct_gap_wo_vs_with"],
            r["pct_cpu_decrease_wo_vs_with"],
        ]
    )
    return row


def save_table3_csv(path: str, table3: Dict, seed: int = None, include_seed: bool = False) -> None:
    import os

    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(_table3_csv_header(include_seed=include_seed))
        for r in table3["rows"]:
            w.writerow(_table3_csv_row(r, seed=seed, include_seed=include_seed))


def run_table3_like_multi_seed(
    instances_root: str,
    instance_names: List[str],
    p: int,
    seeds: Sequence[int],
    k_max: int,
    l_max: int,
    i_max: int,
    strict_feasibility: bool = False,
    final_metric: str = "paper_exact",
    paper_time_limit: float = 300.0,
    paper_mip_gap: float = 0.0,
    paper_threads: int = 0,
    paper_mip_focus: int = 1,
    paper_presolve: int = 2,
    paper_cuts: int = 2,
    paper_symmetry: int = 2,
) -> Dict:
    seed_runs = []
    data_cache: Dict[str, Dict] = {}
    score_cache: Dict[Tuple[str, Tuple[int, ...]], Dict] = {}
    for seed in seeds:
        cfg = build_cfg(
            p,
            int(seed),
            k_max,
            l_max,
            i_max,
            strict_feasibility=bool(strict_feasibility),
        )
        table3 = run_table3_like(
            instances_root,
            instance_names,
            cfg,
            final_metric=final_metric,
            paper_time_limit=paper_time_limit,
            paper_mip_gap=paper_mip_gap,
            paper_threads=paper_threads,
            paper_mip_focus=paper_mip_focus,
            paper_presolve=paper_presolve,
            paper_cuts=paper_cuts,
            paper_symmetry=paper_symmetry,
            data_cache=data_cache,
            score_cache=score_cache,
        )
        seed_runs.append({"seed": int(seed), "table3_like": table3})

    by_instance: Dict[str, List[Tuple[int, Dict]]] = {}
    for run in seed_runs:
        seed = int(run["seed"])
        for row in run["table3_like"]["rows"]:
            by_instance.setdefault(row["instance"], []).append((seed, row))

    best_across_seeds = []
    for inst in sorted(by_instance.keys(), key=lambda x: int(x.replace("Instance", ""))):
        cand = by_instance[inst]
        def _key_with(item: Tuple[int, Dict]) -> float:
            v = _final_obj(item[1]["with_improvement"], final_metric)
            return float(v) if v is not None else float("inf")

        def _key_wo(item: Tuple[int, Dict]) -> float:
            v = _final_obj(item[1]["without_improvement"], final_metric)
            return float(v) if v is not None else float("inf")

        best_with_seed, best_with = min(cand, key=_key_with)
        best_wo_seed, best_wo = min(cand, key=_key_wo)
        best_across_seeds.append(
            {
                "instance": inst,
                "best_with_seed": int(best_with_seed),
                "best_with_improvement": best_with["with_improvement"],
                "best_with_final_obj": _final_obj(best_with["with_improvement"], final_metric),
                "best_without_seed": int(best_wo_seed),
                "best_without_improvement": best_wo["without_improvement"],
                "best_without_final_obj": _final_obj(best_wo["without_improvement"], final_metric),
                "final_metric": final_metric,
            }
        )

    avg_by_seed = [
        {"seed": int(run["seed"]), "average": run["table3_like"]["average"]}
        for run in seed_runs
    ]

    return {
        "seed_runs": seed_runs,
        "best_across_seeds": best_across_seeds,
        "average_by_seed": avg_by_seed,
        "final_metric": final_metric,
    }


def save_table3_multi_seed_csv(path: str, multi_seed: Dict) -> None:
    import os

    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(_table3_csv_header(include_seed=True))
        for run in multi_seed["seed_runs"]:
            seed = int(run["seed"])
            for row in run["table3_like"]["rows"]:
                w.writerow(_table3_csv_row(row, seed=seed, include_seed=True))


def save_table3_best_seed_csv(path: str, multi_seed: Dict) -> None:
    import os

    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "instance",
                "final_metric",
                "best_with_seed",
                "best_with_open_depots_internal",
                "best_with_open_depots",
                "best_with_open_depots_base",
                "best_with_final_obj",
                "best_with_arrival_sim_obj_search",
                "best_with_paper_exact_obj",
                "best_with_paper_exact_status",
                "best_with_runtime_sec",
                "best_without_seed",
                "best_without_open_depots_internal",
                "best_without_open_depots",
                "best_without_open_depots_base",
                "best_without_final_obj",
                "best_without_arrival_sim_obj_search",
                "best_without_paper_exact_obj",
                "best_without_paper_exact_status",
                "best_without_runtime_sec",
            ]
        )
        for r in multi_seed["best_across_seeds"]:
            bw = r["best_with_improvement"]
            bwo = r["best_without_improvement"]
            bw_p = bw.get("paper_exact", {})
            bwo_p = bwo.get("paper_exact", {})
            w.writerow(
                [
                    r["instance"],
                    r.get("final_metric"),
                    r["best_with_seed"],
                    bw["best_x_open"],
                    bw["best_x_open_idx"],
                    bw["best_x_open_base"],
                    r.get("best_with_final_obj"),
                    bw.get("arrival_sim_obj"),
                    bw_p.get("objective"),
                    bw_p.get("status_name"),
                    bw["runtime_sec"],
                    r["best_without_seed"],
                    bwo["best_x_open"],
                    bwo["best_x_open_idx"],
                    bwo["best_x_open_base"],
                    r.get("best_without_final_obj"),
                    bwo.get("arrival_sim_obj"),
                    bwo_p.get("objective"),
                    bwo_p.get("status_name"),
                    bwo["runtime_sec"],
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Table3-like experiment with/without VNS improvement. Open-depot outputs use 0-based candidate indices."
    )
    parser.add_argument("--instances-root", default="./Instance")
    parser.add_argument("--instances", default="1-10", help="Examples: 1-10 | 1,2,3 | Instance3,Instance4")
    parser.add_argument("--p", type=int, default=2)
    parser.add_argument("--seed", type=int, default=-1, help="Single seed; <0 means random seed")
    parser.add_argument("--seeds", default="", help="Optional multi-seed list, e.g. 101,102,103 or 100-105")
    parser.add_argument("--k-max", type=int, default=7)
    parser.add_argument("--l-max", type=int, default=6)
    parser.add_argument("--i-max", type=int, default=5)
    parser.add_argument(
        "--strict-feasibility",
        choices=["true", "false"],
        default="true",
        help="Whether to enforce strict repair/feasibility filtering during VNS.",
    )
    parser.add_argument(
        "--final-metric",
        choices=["paper_exact", "arrival_sim"],
        default="paper_exact",
        help="Search is always by arrival_sim inside heuristic; this controls final ranking/reporting metric.",
    )
    parser.add_argument("--paper-time-limit", type=float, default=300.0, help="Per fixed-combo Gurobi time limit (sec).")
    parser.add_argument("--paper-mip-gap", type=float, default=0.0, help="Per fixed-combo target MIP gap.")
    parser.add_argument("--paper-threads", type=int, default=0, help="Gurobi threads (0=auto).")
    parser.add_argument("--paper-mip-focus", type=int, default=1, help="Gurobi MIPFocus.")
    parser.add_argument("--paper-presolve", type=int, default=2, help="Gurobi Presolve.")
    parser.add_argument("--paper-cuts", type=int, default=2, help="Gurobi Cuts.")
    parser.add_argument("--paper-symmetry", type=int, default=2, help="Gurobi Symmetry.")
    parser.add_argument("--out-json", default="outputs/table3_like_current_code.json")
    parser.add_argument("--out-csv", default="outputs/table3_like_current_code.csv")
    parser.add_argument(
        "--out-best-seed-csv",
        default="outputs/table3_like_current_code_best_across_seeds.csv",
        help="Only used when --seeds has multiple values.",
    )
    args = parser.parse_args()

    instance_names = parse_instance_names(args.instances)
    base_seed = int(args.seed)
    if base_seed < 0:
        base_seed = int(random.SystemRandom().randint(1, 10**9 - 1))
        print(f"Using random seed: {base_seed}")
    seeds = parse_seeds(args.seeds, base_seed)
    strict_feas = args.strict_feasibility.lower() == "true"

    if len(seeds) == 1:
        cfg = build_cfg(
            args.p,
            seeds[0],
            args.k_max,
            args.l_max,
            args.i_max,
            strict_feasibility=strict_feas,
        )
        table3 = run_table3_like(
            args.instances_root,
            instance_names,
            cfg,
            final_metric=args.final_metric,
            paper_time_limit=args.paper_time_limit,
            paper_mip_gap=args.paper_mip_gap,
            paper_threads=args.paper_threads,
            paper_mip_focus=args.paper_mip_focus,
            paper_presolve=args.paper_presolve,
            paper_cuts=args.paper_cuts,
            paper_symmetry=args.paper_symmetry,
        )
        payload = {
            "config": {
                "instances": instance_names,
                "p": args.p,
                "seed": seeds[0],
                "k_max": args.k_max,
                "l_max": args.l_max,
                "i_max": args.i_max,
                "strict_feasibility": strict_feas,
                "final_metric": args.final_metric,
                "paper_eval": {
                    "time_limit_sec": args.paper_time_limit,
                    "mip_gap": args.paper_mip_gap,
                    "threads": args.paper_threads,
                    "mip_focus": args.paper_mip_focus,
                    "presolve": args.paper_presolve,
                    "cuts": args.paper_cuts,
                    "symmetry": args.paper_symmetry,
                },
                "open_depots_indexing": "0-based index in candidate_depots (W-order)",
            },
            "table3_like": table3,
        }
        save_json(args.out_json, payload)
        save_table3_csv(args.out_csv, table3)

        avg = table3["average"]
        print(f"Saved JSON: {args.out_json}")
        print(f"Saved CSV : {args.out_csv}")
        print(
            "Average -> metric={} with_final={}, with_arrival={:.4f}, with_cpu={:.3f}s, wo_final={}, wo_arrival={:.4f}, wo_cpu={:.3f}s, gap={}, cpu_dec={}".format(
                avg.get("final_metric"),
                avg.get("with_improvement_final_obj"),
                avg["with_improvement_arrival_sim_obj"],
                avg["with_improvement_runtime_sec"],
                avg.get("without_improvement_final_obj"),
                avg["without_improvement_arrival_sim_obj"],
                avg["without_improvement_runtime_sec"],
                avg.get("avg_pct_gap_wo_vs_with"),
                avg.get("avg_pct_cpu_decrease_wo_vs_with"),
            )
        )
        return

    multi = run_table3_like_multi_seed(
        instances_root=args.instances_root,
        instance_names=instance_names,
        p=args.p,
        seeds=seeds,
        k_max=args.k_max,
        l_max=args.l_max,
        i_max=args.i_max,
        strict_feasibility=strict_feas,
        final_metric=args.final_metric,
        paper_time_limit=args.paper_time_limit,
        paper_mip_gap=args.paper_mip_gap,
        paper_threads=args.paper_threads,
        paper_mip_focus=args.paper_mip_focus,
        paper_presolve=args.paper_presolve,
        paper_cuts=args.paper_cuts,
        paper_symmetry=args.paper_symmetry,
    )
    payload = {
        "config": {
            "instances": instance_names,
            "p": args.p,
            "seeds": list(seeds),
            "k_max": args.k_max,
            "l_max": args.l_max,
            "i_max": args.i_max,
            "strict_feasibility": strict_feas,
            "final_metric": args.final_metric,
            "paper_eval": {
                "time_limit_sec": args.paper_time_limit,
                "mip_gap": args.paper_mip_gap,
                "threads": args.paper_threads,
                "mip_focus": args.paper_mip_focus,
                "presolve": args.paper_presolve,
                "cuts": args.paper_cuts,
                "symmetry": args.paper_symmetry,
            },
            "open_depots_indexing": "0-based index in candidate_depots (W-order)",
        },
        "table3_like_multi_seed": multi,
    }
    save_json(args.out_json, payload)
    save_table3_multi_seed_csv(args.out_csv, multi)
    save_table3_best_seed_csv(args.out_best_seed_csv, multi)

    print(f"Saved JSON            : {args.out_json}")
    print(f"Saved seed-detail CSV : {args.out_csv}")
    print(f"Saved best-seed CSV   : {args.out_best_seed_csv}")
    for rec in multi["average_by_seed"]:
        avg = rec["average"]
        print(
            "Seed {} -> metric={} with_final={}, with_arrival={:.4f}, with_cpu={:.3f}s, wo_final={}, wo_arrival={:.4f}, wo_cpu={:.3f}s, gap={}, cpu_dec={}".format(
                rec["seed"],
                avg.get("final_metric"),
                avg.get("with_improvement_final_obj"),
                avg["with_improvement_arrival_sim_obj"],
                avg["with_improvement_runtime_sec"],
                avg.get("without_improvement_final_obj"),
                avg["without_improvement_arrival_sim_obj"],
                avg["without_improvement_runtime_sec"],
                avg.get("avg_pct_gap_wo_vs_with"),
                avg.get("avg_pct_cpu_decrease_wo_vs_with"),
            )
        )


if __name__ == "__main__":
    main()
