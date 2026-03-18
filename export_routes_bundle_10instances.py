import argparse
import ast
import csv
import json
import os
import random
from typing import Dict, List, Sequence, Tuple

from alg import HeuristicConfig
from gurobi_exact_small_enumeration import _prepare_data, solve_fixed_combo_commodity
from main import load_instance_data
from paper_eval_common import parse_instance_names
from run_experiment import _normalize_weights, _run_fixed_x


def _parse_list_cell(raw: str) -> List[int]:
    text = (raw or "").strip()
    if not text:
        return []
    try:
        arr = ast.literal_eval(text)
        return [int(x) for x in arr]
    except Exception as exc:
        raise ValueError(f"Cannot parse list cell: {raw!r}") from exc


def _idx_map(candidates: Sequence[int]) -> Dict[int, int]:
    return {int(k): int(i) for i, k in enumerate(candidates)}


def _save_json(path: str, obj: Dict) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _run_table3_fixedx_paths(
    *,
    instance_name: str,
    instances_root: str,
    open_depots: Sequence[int],
    k_max: int,
    l_max: int,
    i_max: int,
    seed: int,
) -> Dict:
    folder = os.path.join(instances_root, instance_name)
    base_instance, scenarios = load_instance_data(folder)
    weights = _normalize_weights([], len(scenarios))
    cfg = HeuristicConfig(
        num_depots_to_open=len(open_depots),
        num_scenarios=len(scenarios),
        seed=int(seed),
        k_max=int(k_max),
        l_max=int(l_max),
        i_max=int(i_max),
        drone_time_is_roundtrip=True,
        normalize_by_num_demands=True,
        strict_feasibility=True,
    )
    depot_base_map = dict(getattr(base_instance, "depot_base_map", {}))
    return _run_fixed_x(
        instance_name=instance_name,
        depot_base_map=depot_base_map,
        scenarios=scenarios,
        weights=weights,
        x_open=set(int(x) for x in open_depots),
        cfg=cfg,
    )


def _run_gurobi_routes(
    *,
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
) -> Dict:
    base, scenarios, demand_nodes, candidates = _prepare_data(
        instance_name=instance_name,
        instances_root=instances_root,
        max_scenarios=0,
        max_demands=0,
    )
    pos = _idx_map(candidates)
    depot_base_map = dict(getattr(base, "depot_base_map", {}))
    combo = sorted(int(x) for x in open_depots)

    solved = solve_fixed_combo_commodity(
        instance_name=instance_name,
        scenarios=scenarios,
        demand_nodes=demand_nodes,
        open_depots=combo,
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
        return_solution_details=True,
        fixed_binary_values=None,
    )

    return {
        "method": "gurobi_paper_exact_fixed_combo",
        "instance": instance_name,
        "open_depots_internal": combo,
        "open_depots_idx": sorted(pos[int(k)] for k in combo if int(k) in pos),
        "open_depots_base": [depot_base_map.get(int(k), int(k)) for k in combo],
        "result": solved,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export route bundle for 10 instances: Table3 fixed-x paths and Gurobi route details."
    )
    parser.add_argument("--compare-csv", default="outputs/compare_table3_paper_unified_vs_gurobi.csv")
    parser.add_argument("--instances-root", default="./Instance")
    parser.add_argument("--out-dir", default="outputs/routes_bundle_10instances")
    parser.add_argument("--instances", default="1-10", help="Examples: 1-5 | 1,2,3 | Instance1,Instance2")
    parser.add_argument("--table3-k-max", type=int, default=7)
    parser.add_argument("--table3-l-max", type=int, default=8)
    parser.add_argument("--table3-i-max", type=int, default=7)
    parser.add_argument("--table3-seed", type=int, default=-1, help="<0 means random seed")
    parser.add_argument("--gurobi-time-limit", type=float, default=1200.0)
    parser.add_argument("--gurobi-mip-gap", type=float, default=0.0)
    parser.add_argument("--gurobi-threads", type=int, default=0)
    parser.add_argument("--gurobi-mip-focus", type=int, default=1)
    parser.add_argument("--gurobi-presolve", type=int, default=2)
    parser.add_argument("--gurobi-cuts", type=int, default=2)
    parser.add_argument("--gurobi-symmetry", type=int, default=2)
    parser.add_argument(
        "--include-gurobi-same-as-table3",
        action="store_true",
        help="Also export Gurobi routes for Table3's with-combo (in addition to Gurobi best-combo).",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rows = list(csv.DictReader(open(args.compare_csv, "r", encoding="utf-8")))
    if not rows:
        raise ValueError(f"No rows in compare csv: {args.compare_csv}")

    selected_instances = set(parse_instance_names(args.instances))
    rows = [r for r in rows if str(r.get("instance", "")).strip() in selected_instances]
    if not rows:
        raise ValueError(f"No matching rows for instances={args.instances} in {args.compare_csv}")

    table3_seed = int(args.table3_seed)
    if table3_seed < 0:
        table3_seed = int(random.SystemRandom().randint(1, 10**9 - 1))
        print(f"Using random table3 seed: {table3_seed}")

    gurobi_cache: Dict[Tuple[str, Tuple[int, ...]], Dict] = {}
    index_rows: List[Dict] = []

    for row in rows:
        instance = str(row["instance"]).strip()
        table3_combo = sorted(_parse_list_cell(row["table3_with_internal"]))
        gurobi_best_combo = sorted(_parse_list_cell(row["gurobi_best_internal"]))

        # Table3 fixed-x paths
        t3_obj = _run_table3_fixedx_paths(
            instance_name=instance,
            instances_root=args.instances_root,
            open_depots=table3_combo,
            k_max=args.table3_k_max,
            l_max=args.table3_l_max,
            i_max=args.table3_i_max,
            seed=table3_seed,
        )
        t3_path = os.path.join(
            args.out_dir,
            f"{instance}_table3_fixedx_combo_{'_'.join(str(x) for x in table3_combo)}.json",
        )
        _save_json(t3_path, t3_obj)

        # Gurobi best-combo routes
        best_key = (instance, tuple(gurobi_best_combo))
        if best_key not in gurobi_cache:
            gurobi_cache[best_key] = _run_gurobi_routes(
                instance_name=instance,
                instances_root=args.instances_root,
                open_depots=gurobi_best_combo,
                time_limit=args.gurobi_time_limit,
                mip_gap=args.gurobi_mip_gap,
                threads=args.gurobi_threads,
                mip_focus=args.gurobi_mip_focus,
                presolve=args.gurobi_presolve,
                cuts=args.gurobi_cuts,
                symmetry=args.gurobi_symmetry,
            )
        g_best_path = os.path.join(
            args.out_dir,
            f"{instance}_gurobi_best_combo_{'_'.join(str(x) for x in gurobi_best_combo)}.json",
        )
        _save_json(g_best_path, gurobi_cache[best_key])

        g_same_path = ""
        if args.include_gurobi_same_as_table3:
            same_key = (instance, tuple(table3_combo))
            if same_key not in gurobi_cache:
                gurobi_cache[same_key] = _run_gurobi_routes(
                    instance_name=instance,
                    instances_root=args.instances_root,
                    open_depots=table3_combo,
                    time_limit=args.gurobi_time_limit,
                    mip_gap=args.gurobi_mip_gap,
                    threads=args.gurobi_threads,
                    mip_focus=args.gurobi_mip_focus,
                    presolve=args.gurobi_presolve,
                    cuts=args.gurobi_cuts,
                    symmetry=args.gurobi_symmetry,
                )
            g_same_path = os.path.join(
                args.out_dir,
                f"{instance}_gurobi_same_as_table3_combo_{'_'.join(str(x) for x in table3_combo)}.json",
            )
            _save_json(g_same_path, gurobi_cache[same_key])

        index_rows.append(
            {
                "instance": instance,
                "table3_combo_internal": table3_combo,
                "gurobi_best_combo_internal": gurobi_best_combo,
                "table3_paths_json": t3_path,
                "gurobi_best_paths_json": g_best_path,
                "gurobi_same_as_table3_paths_json": g_same_path,
            }
        )
        print(f"[done] {instance}")

    index_json = os.path.join(args.out_dir, "route_bundle_index.json")
    _save_json(index_json, {"rows": index_rows, "count": len(index_rows)})

    index_csv = os.path.join(args.out_dir, "route_bundle_index.csv")
    with open(index_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "instance",
                "table3_combo_internal",
                "gurobi_best_combo_internal",
                "table3_paths_json",
                "gurobi_best_paths_json",
                "gurobi_same_as_table3_paths_json",
            ]
        )
        for r in index_rows:
            w.writerow(
                [
                    r["instance"],
                    r["table3_combo_internal"],
                    r["gurobi_best_combo_internal"],
                    r["table3_paths_json"],
                    r["gurobi_best_paths_json"],
                    r["gurobi_same_as_table3_paths_json"],
                ]
            )

    print(f"Saved route bundle index JSON: {index_json}")
    print(f"Saved route bundle index CSV : {index_csv}")


if __name__ == "__main__":
    main()
