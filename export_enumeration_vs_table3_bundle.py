import argparse
import csv
import json
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from compare_unified_scoring import _build_strict_fixed_from_table3_paths, _repair_table3_paths_for_paper_exact
from gurobi_exact_small_enumeration import _prepare_data, solve_fixed_combo_commodity
from paper_eval_common import parse_instance_names


def _ensure_parent(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _save_json(path: str, obj: Dict) -> None:
    _ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _to_int(x: Any) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None


def _node_to_internal(node: Any, idx_to_internal: Dict[int, int]) -> Optional[int]:
    if isinstance(node, str) and node.startswith("D"):
        idx = _to_int(node[1:])
        if idx is None:
            return None
        return int(idx_to_internal.get(int(idx), int(idx)))
    iv = _to_int(node)
    if iv is None:
        return None
    return int(iv)


def _depot_key_to_internal(key: Any, idx_to_internal: Dict[int, int], internal_set: set) -> Optional[int]:
    ik = _to_int(key)
    if ik is None:
        return None
    if ik in internal_set:
        return int(ik)
    if ik in idx_to_internal:
        return int(idx_to_internal[ik])
    return int(ik)


def _extract_table3_paths_internal(
    payload: Dict,
) -> Tuple[List[Dict[int, List[int]]], List[Dict[int, List[int]]], List[int]]:
    if "x_open_internal" in payload:
        open_internal = [int(x) for x in payload.get("x_open_internal", [])]
    else:
        open_internal = [int(x) for x in payload.get("x_open", [])]
    internal_set = set(open_internal)

    idx_to_internal_raw = payload.get("depot_idx_to_internal", {})
    idx_to_internal = {int(k): int(v) for k, v in idx_to_internal_raw.items()}

    tt_by_s: List[Dict[int, List[int]]] = []
    ds_by_s: List[Dict[int, List[int]]] = []
    for s in payload.get("scenarios", []):
        tt_raw = s.get("tt", {})
        ds_raw = s.get("ds", {})
        tt: Dict[int, List[int]] = {}
        ds: Dict[int, List[int]] = {}

        for k, walk in tt_raw.items():
            kk = _depot_key_to_internal(k, idx_to_internal, internal_set)
            if kk is None:
                continue
            parsed = []
            for n in walk or []:
                iv = _node_to_internal(n, idx_to_internal)
                if iv is None:
                    continue
                parsed.append(int(iv))
            tt[int(kk)] = parsed

        for k, seq in ds_raw.items():
            kk = _depot_key_to_internal(k, idx_to_internal, internal_set)
            if kk is None:
                continue
            ds[int(kk)] = [int(v) for v in (seq or [])]

        tt_by_s.append(tt)
        ds_by_s.append(ds)

    return tt_by_s, ds_by_s, sorted(int(x) for x in open_internal)


def _collect_table3_paths(bundle_dir: str) -> Dict[str, str]:
    by_inst: Dict[str, str] = {}
    for fn in os.listdir(bundle_dir):
        if not fn.endswith(".json"):
            continue
        m = re.match(r"^(Instance\d+)_table3_fixedx_combo_.*\.json$", fn)
        if m:
            by_inst[m.group(1)] = os.path.join(bundle_dir, fn)
    return by_inst


def _load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _pick_best_row(enum_payload: Dict) -> Dict:
    rows = list(enum_payload.get("combination_results", []))
    for row in rows:
        if int(row.get("is_best", 0)) == 1:
            return row

    ranked = [r for r in rows if r.get("objective") is not None]
    if ranked:
        ranked.sort(key=lambda r: float(r.get("rank_by_objective", 10**9)))
        return ranked[0]

    raise ValueError(f"No feasible combination found for {enum_payload.get('instance')}")


def _build_best_json(enum_payload: Dict, best_row: Dict, enum_path: str) -> Dict:
    return {
        "method": "gurobi_paper_exact_enumeration_best_combo",
        "instance": enum_payload.get("instance"),
        "selection_metric": enum_payload.get("summary", {}).get("selection_metric"),
        "from_enumeration_json": enum_path,
        "open_depots_internal": list(best_row.get("open_depots", [])),
        "open_depots_idx": list(best_row.get("open_depots_idx", [])),
        "open_depots_base": list(best_row.get("open_depots_base", [])),
        "result": best_row,
        "instance_summary": enum_payload.get("summary", {}),
    }


def _flatten_gurobi_best_paths(instance: str, best_json: Dict) -> List[Dict]:
    out: List[Dict] = []
    rows = (
        best_json.get("result", {})
        .get("solution_details", {})
        .get("scenario_routes", [])
    )
    for scen in rows:
        s = int(scen.get("scenario_index", 0))
        for dep in scen.get("depots", []):
            out.append(
                {
                    "instance": instance,
                    "source": "gurobi_best",
                    "scenario_index": int(s),
                    "depot_internal": int(dep.get("depot_internal")),
                    "truck_walk_eval_preferred": json.dumps(dep.get("truck_walk_eval_preferred", []), ensure_ascii=False),
                    "truck_assigned_demands": json.dumps(dep.get("truck_assigned_demands", []), ensure_ascii=False),
                    "drone_sequence_spt": json.dumps(dep.get("drone_sequence_spt", []), ensure_ascii=False),
                    "drone_assigned_demands": json.dumps(dep.get("drone_assigned_demands", []), ensure_ascii=False),
                    "drone_arrival_time": json.dumps(dep.get("drone_arrival_time", {}), ensure_ascii=False, sort_keys=True),
                }
            )
    return out


def _flatten_table3_paths(instance: str, table3_payload: Dict) -> List[Dict]:
    tt_by_s, ds_by_s, open_internal = _extract_table3_paths_internal(table3_payload)
    source = "table3_paper_exact_repaired" if bool(table3_payload.get("paper_exact_repaired")) else "table3_existing"
    out: List[Dict] = []
    for s, tt in enumerate(tt_by_s):
        ds = ds_by_s[s] if s < len(ds_by_s) else {}
        for k in open_internal:
            out.append(
                {
                    "instance": instance,
                    "source": source,
                    "scenario_index": int(s),
                    "depot_internal": int(k),
                    "truck_walk_eval_preferred": json.dumps(tt.get(int(k), []), ensure_ascii=False),
                    "truck_assigned_demands": "",
                    "drone_sequence_spt": json.dumps(ds.get(int(k), []), ensure_ascii=False),
                    "drone_assigned_demands": "",
                    "drone_arrival_time": "",
                }
            )
    return out


def _build_repaired_table3_payload(
    original_payload: Dict,
    repaired_tt_by_s: Sequence[Dict[int, List[int]]],
    repaired_ds_by_s: Sequence[Dict[int, List[int]]],
    diagnostics: Sequence[Dict[str, object]],
    paper_exact_fixed_obj: Optional[float],
    paper_exact_fixed_status: str,
) -> Dict:
    out = dict(original_payload)
    out["paper_exact_repaired"] = True
    out["paper_exact_fixed_obj"] = paper_exact_fixed_obj
    out["paper_exact_fixed_status"] = paper_exact_fixed_status
    out["paper_exact_repair_diagnostics"] = list(diagnostics)

    repaired_rows: List[Dict[str, object]] = []
    original_rows = list(original_payload.get("scenarios", []))
    for s, row in enumerate(original_rows):
        base_row = dict(row)
        tt = repaired_tt_by_s[s] if s < len(repaired_tt_by_s) else {}
        ds = repaired_ds_by_s[s] if s < len(repaired_ds_by_s) else {}
        base_row["tt"] = {str(int(k)): [int(x) for x in walk] for k, walk in sorted(tt.items())}
        base_row["ds"] = {str(int(k)): [int(x) for x in seq] for k, seq in sorted(ds.items())}
        repaired_rows.append(base_row)
    out["scenarios"] = repaired_rows
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export best paths from enumeration outputs and compare them with existing Table3 route JSONs."
    )
    parser.add_argument("--enum-per-instance-dir", required=True)
    parser.add_argument("--table3-bundle-dir", default="outputs/routes_bundle_10instances")
    parser.add_argument("--instances-root", default="./Instance")
    parser.add_argument("--instances", default="1-10")
    parser.add_argument("--time-limit", type=float, default=100.0)
    parser.add_argument("--mip-gap", type=float, default=0.0)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--mip-focus", type=int, default=1)
    parser.add_argument("--presolve", type=int, default=2)
    parser.add_argument("--cuts", type=int, default=2)
    parser.add_argument("--symmetry", type=int, default=2)
    parser.add_argument("--out-dir", default="")
    args = parser.parse_args()

    instances = parse_instance_names(args.instances)
    table3_by_inst = _collect_table3_paths(args.table3_bundle_dir)

    out_dir = args.out_dir or os.path.join(args.enum_per_instance_dir, "..", "comparison_bundle_vs_table3")
    out_dir = os.path.normpath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    summary_rows: List[Dict] = []
    bundle_index_rows: List[Dict] = []
    gurobi_flat_rows: List[Dict] = []
    table3_flat_rows: List[Dict] = []

    for inst in instances:
        enum_path = os.path.join(args.enum_per_instance_dir, f"{inst}.json")
        table3_path = table3_by_inst.get(inst)
        if not os.path.exists(enum_path):
            raise FileNotFoundError(f"Enumeration JSON not found: {enum_path}")
        if not table3_path or not os.path.exists(table3_path):
            raise FileNotFoundError(f"Table3 path JSON not found for {inst}")

        enum_payload = _load_json(enum_path)
        table3_payload = _load_json(table3_path)
        best_row = _pick_best_row(enum_payload)
        best_json = _build_best_json(enum_payload, best_row, enum_path)

        combo_suffix = "_".join(str(x) for x in best_json["open_depots_internal"])
        best_json_path = os.path.join(out_dir, f"{inst}_gurobi_best_combo_{combo_suffix}.json")
        _save_json(best_json_path, best_json)

        base, scenarios, demand_nodes, candidates = _prepare_data(inst, args.instances_root, 0, 0)
        depot_base_map = {int(k): int(v) for k, v in dict(getattr(base, "depot_base_map", {})).items()}
        candidate_pos = {int(k): int(i) for i, k in enumerate(candidates)}

        tt_by_s, ds_by_s, table3_open_internal = _extract_table3_paths_internal(table3_payload)
        repaired_tt_by_s, repaired_ds_by_s, repair_diag = _repair_table3_paths_for_paper_exact(
            scenarios=scenarios,
            demand_nodes=demand_nodes,
            open_depots=table3_open_internal,
            tt_by_s=tt_by_s,
            ds_by_s=ds_by_s,
        )
        fixed_t3 = _build_strict_fixed_from_table3_paths(
            scenarios=scenarios,
            demand_nodes=demand_nodes,
            open_depots=table3_open_internal,
            tt_by_s=repaired_tt_by_s,
            ds_by_s=repaired_ds_by_s,
        )

        table3_eval = solve_fixed_combo_commodity(
            instance_name=inst,
            scenarios=scenarios,
            demand_nodes=demand_nodes,
            open_depots=table3_open_internal,
            time_limit=float(args.time_limit),
            mip_gap=float(args.mip_gap),
            mip_gap_abs=None,
            threads=int(args.threads),
            output_flag=0,
            mip_focus=int(args.mip_focus),
            heuristics=-1.0,
            presolve=int(args.presolve),
            cuts=int(args.cuts),
            symmetry=int(args.symmetry),
            return_solution_details=False,
            fixed_binary_values=fixed_t3,
        )

        repaired_combo_suffix = "_".join(str(x) for x in table3_open_internal)
        repaired_table3_json_path = os.path.join(
            out_dir,
            f"{inst}_table3_fixedx_combo_{repaired_combo_suffix}_paper_exact_repaired.json",
        )
        repaired_table3_payload = _build_repaired_table3_payload(
            original_payload=table3_payload,
            repaired_tt_by_s=repaired_tt_by_s,
            repaired_ds_by_s=repaired_ds_by_s,
            diagnostics=repair_diag,
            paper_exact_fixed_obj=table3_eval.get("objective"),
            paper_exact_fixed_status=str(table3_eval.get("status_name", "")),
        )
        _save_json(repaired_table3_json_path, repaired_table3_payload)

        paper_t3 = table3_eval.get("objective")
        paper_best = best_row.get("objective")
        gap = None
        if paper_t3 is not None and paper_best is not None:
            gap = float(paper_t3) - float(paper_best)

        summary_rows.append(
            {
                "instance": inst,
                "table3_combo_internal": list(table3_open_internal),
                "table3_combo_idx": sorted(candidate_pos[int(k)] for k in table3_open_internal if int(k) in candidate_pos),
                "table3_combo_base": [int(depot_base_map.get(int(k), int(k))) for k in table3_open_internal],
                "table3_arrival_obj_reference": table3_payload.get("expected_obj_a5"),
                "paper_obj_table3_path_fixed": paper_t3,
                "paper_status_table3_path_fixed": table3_eval.get("status_name"),
                "gurobi_best_combo_internal": list(best_json.get("open_depots_internal", [])),
                "gurobi_best_combo_idx": list(best_json.get("open_depots_idx", [])),
                "gurobi_best_combo_base": list(best_json.get("open_depots_base", [])),
                "paper_obj_gurobi_best_combo_opt": paper_best,
                "gurobi_best_status": best_row.get("status_name"),
                "gurobi_best_runtime_sec": best_row.get("runtime_sec"),
                "gurobi_best_bound": best_row.get("best_bound"),
                "gurobi_best_achieved_mip_gap_pct": best_row.get("achieved_mip_gap_pct"),
                "instance_optimal_combo_count": enum_payload.get("summary", {}).get("optimal_count"),
                "instance_time_limit_combo_count": enum_payload.get("summary", {}).get("time_limit_count"),
                "gap_table3_path_minus_gurobi_best_combo_opt": gap,
                "match_table3_combo_vs_gurobi_best_combo": (
                    list(table3_open_internal) == list(best_json.get("open_depots_internal", []))
                ),
                "table3_json": table3_path,
                "table3_repaired_json": repaired_table3_json_path,
                "gurobi_best_json": best_json_path,
            }
        )

        bundle_index_rows.append(
            {
                "instance": inst,
                "table3_combo_internal": list(table3_open_internal),
                "gurobi_best_combo_internal": list(best_json.get("open_depots_internal", [])),
                "table3_paths_json": table3_path,
                "table3_repaired_paths_json": repaired_table3_json_path,
                "gurobi_best_paths_json": best_json_path,
            }
        )

        gurobi_flat_rows.extend(_flatten_gurobi_best_paths(inst, best_json))
        table3_flat_rows.extend(_flatten_table3_paths(inst, table3_payload))
        table3_flat_rows.extend(_flatten_table3_paths(inst, repaired_table3_payload))
        print(f"[done] {inst}")

    if not summary_rows:
        raise ValueError("No summary rows generated.")

    summary_csv = os.path.join(out_dir, "comparison_summary.csv")
    summary_json = os.path.join(out_dir, "comparison_summary.json")
    bundle_index_csv = os.path.join(out_dir, "route_bundle_index.csv")
    bundle_index_json = os.path.join(out_dir, "route_bundle_index.json")
    gurobi_flat_csv = os.path.join(out_dir, "gurobi_best_paths_flat.csv")
    table3_flat_csv = os.path.join(out_dir, "table3_paths_flat.csv")

    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    _save_json(summary_json, {"rows": summary_rows, "count": len(summary_rows)})

    with open(bundle_index_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(bundle_index_rows[0].keys()))
        w.writeheader()
        w.writerows(bundle_index_rows)
    _save_json(bundle_index_json, {"rows": bundle_index_rows, "count": len(bundle_index_rows)})

    with open(gurobi_flat_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(gurobi_flat_rows[0].keys()))
        w.writeheader()
        w.writerows(gurobi_flat_rows)

    with open(table3_flat_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(table3_flat_rows[0].keys()))
        w.writeheader()
        w.writerows(table3_flat_rows)

    print(f"Saved summary CSV: {summary_csv}")
    print(f"Saved summary JSON: {summary_json}")
    print(f"Saved route bundle index CSV: {bundle_index_csv}")
    print(f"Saved route bundle index JSON: {bundle_index_json}")
    print(f"Saved Gurobi best paths flat CSV: {gurobi_flat_csv}")
    print(f"Saved Table3 paths flat CSV: {table3_flat_csv}")


if __name__ == "__main__":
    main()
