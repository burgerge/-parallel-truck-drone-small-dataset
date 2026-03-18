import argparse
import ast
import csv
import json
import os
from typing import Any, Dict, List, Optional, Tuple

from gurobi_exact_small_enumeration import _prepare_data


def _ensure_parent(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _save_json(path: str, obj: Dict) -> None:
    _ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _parse_list_cell(raw: str) -> List[int]:
    text = (raw or "").strip()
    if not text:
        return []
    vals = ast.literal_eval(text)
    return [int(x) for x in vals]


def _to_float(raw: Any) -> Optional[float]:
    if raw in (None, ""):
        return None
    return float(raw)


def _depot_label(node: int, internal_to_idx: Dict[int, int]) -> Any:
    node = int(node)
    if node in internal_to_idx:
        return f"D{internal_to_idx[node]}"
    return int(node)


def _encode_walk(walk: List[int], internal_to_idx: Dict[int, int]) -> List[Any]:
    return [_depot_label(int(node), internal_to_idx) for node in (walk or [])]


def _table3_depot_key_to_idx(dep_key: Any, internal_to_idx: Dict[int, int], candidate_count: int) -> int:
    dep = int(dep_key)
    if dep in internal_to_idx:
        return int(internal_to_idx[dep])
    if 0 <= dep < int(candidate_count):
        return int(dep)
    raise KeyError(dep)


def _table3_node_to_encoded(node: Any, internal_to_idx: Dict[int, int]) -> Any:
    if isinstance(node, str) and node.startswith("D"):
        return f"D{int(node[1:])}"
    iv = int(node)
    return _depot_label(iv, internal_to_idx)


def _collect_table3_scenarios(
    table3_payload: Dict,
    internal_to_idx: Dict[int, int],
    candidate_count: int,
) -> List[Dict]:
    out: List[Dict] = []
    for scen in table3_payload.get("scenarios", []):
        depots: List[Dict] = []
        tt = scen.get("tt", {})
        ds = scen.get("ds", {})
        all_depots = sorted(set(list(tt.keys()) + list(ds.keys())), key=lambda x: int(x))
        for dep in all_depots:
            dep_idx = _table3_depot_key_to_idx(dep, internal_to_idx, candidate_count)
            dep_internal = next((int(k) for k, v in internal_to_idx.items() if int(v) == int(dep_idx)), None)
            depots.append(
                {
                    "depot_internal": dep_internal,
                    "depot_idx": int(dep_idx),
                    "truck_path_0_4": [_table3_node_to_encoded(x, internal_to_idx) for x in tt.get(dep, [])],
                    "drone_seq": [int(x) for x in ds.get(dep, [])],
                }
            )
        out.append(
            {
                "scenario_index": int(scen.get("scenario_index", 0)),
                "depots": depots,
            }
        )
    return out


def _collect_gurobi_scenarios(gurobi_payload: Dict, internal_to_idx: Dict[int, int]) -> List[Dict]:
    rows = (
        gurobi_payload.get("result", {})
        .get("solution_details", {})
        .get("scenario_routes", [])
    )
    out: List[Dict] = []
    for scen in rows:
        depots: List[Dict] = []
        for dep in scen.get("depots", []):
            dep_internal = int(dep.get("depot_internal"))
            depots.append(
                {
                    "depot_internal": dep_internal,
                    "depot_idx": int(internal_to_idx[dep_internal]),
                    "truck_path_0_4": _encode_walk(
                        [int(x) for x in dep.get("truck_walk_eval_preferred", [])],
                        internal_to_idx,
                    ),
                    "drone_seq": [int(x) for x in dep.get("drone_sequence_spt", [])],
                }
            )
        out.append(
            {
                "scenario_index": int(scen.get("scenario_index", 0)),
                "depots": depots,
            }
        )
    return out


def _scenario_depot_map(rows: List[Dict]) -> Dict[Tuple[int, int], Dict]:
    out: Dict[Tuple[int, int], Dict] = {}
    for scen in rows:
        s = int(scen.get("scenario_index", 0))
        for dep in scen.get("depots", []):
            k = int(dep.get("depot_idx"))
            out[(int(s), int(k))] = dict(dep)
    return out


def _rows_by_instance(rows: List[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    return {str(r.get("instance", "")).strip(): dict(r) for r in rows}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build detailed Table3-vs-Gurobi comparison tables with depot idx 0-4 path encoding."
    )
    parser.add_argument(
        "--comparison-summary-csv",
        default="outputs/gurobi_enum_paper_exact_100s_20260318_195048/comparison_bundle_vs_table3/comparison_summary.csv",
    )
    parser.add_argument(
        "--table3-unified-csv",
        default="outputs/table3_like_paper_exact_unified.csv",
    )
    parser.add_argument("--instances-root", default="./Instance")
    parser.add_argument("--out-dir", default="")
    args = parser.parse_args()

    comparison_rows = _load_csv_rows(args.comparison_summary_csv)
    unified_rows = _load_csv_rows(args.table3_unified_csv)
    unified_by_inst = _rows_by_instance(unified_rows)

    out_dir = args.out_dir or os.path.dirname(args.comparison_summary_csv)
    os.makedirs(out_dir, exist_ok=True)

    instance_rows: List[Dict[str, Any]] = []
    route_rows: List[Dict[str, Any]] = []

    for cmp_row in comparison_rows:
        inst = str(cmp_row.get("instance", "")).strip()
        table3_json_path = str(cmp_row.get("table3_json", "")).strip()
        table3_repaired_json_path = str(cmp_row.get("table3_repaired_json", "")).strip()
        gurobi_json_path = str(cmp_row.get("gurobi_best_json", "")).strip()
        if not table3_json_path or not gurobi_json_path:
            continue

        table3_payload = _load_json(table3_json_path)
        table3_repaired_payload = (
            _load_json(table3_repaired_json_path)
            if table3_repaired_json_path and os.path.exists(table3_repaired_json_path)
            else table3_payload
        )
        gurobi_payload = _load_json(gurobi_json_path)

        base, _scenarios, _demand_nodes, candidates = _prepare_data(inst, args.instances_root, 0, 0)
        internal_to_idx = {int(k): int(i) for i, k in enumerate(candidates)}

        table3_raw_scenarios = _collect_table3_scenarios(table3_payload, internal_to_idx, len(candidates))
        table3_scenarios = _collect_table3_scenarios(table3_repaired_payload, internal_to_idx, len(candidates))
        gurobi_scenarios = _collect_gurobi_scenarios(gurobi_payload, internal_to_idx)
        table3_map = _scenario_depot_map(table3_scenarios)
        table3_raw_map = _scenario_depot_map(table3_raw_scenarios)
        gurobi_map = _scenario_depot_map(gurobi_scenarios)

        unified = unified_by_inst.get(inst, {})

        instance_rows.append(
            {
                "instance": inst,
                "table3_combo_idx": _parse_list_cell(str(cmp_row.get("table3_combo_idx", ""))),
                "table3_arrival_obj_reference_a5": _to_float(table3_payload.get("expected_obj_a5")),
                "table3_repo_final_obj": _to_float(unified.get("with_final_obj")),
                "table3_repo_runtime_sec": _to_float(unified.get("with_runtime_sec")),
                "table3_fixed_path_paper_obj": _to_float(cmp_row.get("paper_obj_table3_path_fixed")),
                "table3_fixed_path_status": cmp_row.get("paper_status_table3_path_fixed"),
                "table3_paths_0_4_json": json.dumps(table3_scenarios, ensure_ascii=False),
                "table3_paths_raw_0_4_json": json.dumps(table3_raw_scenarios, ensure_ascii=False),
                "gurobi_combo_idx": _parse_list_cell(str(cmp_row.get("gurobi_best_combo_idx", ""))),
                "gurobi_runtime_sec": _to_float(cmp_row.get("gurobi_best_runtime_sec")),
                "gurobi_paper_obj": _to_float(cmp_row.get("paper_obj_gurobi_best_combo_opt")),
                "gurobi_status": cmp_row.get("gurobi_best_status"),
                "gurobi_paths_0_4_json": json.dumps(gurobi_scenarios, ensure_ascii=False),
                "obj_gap_table3_minus_gurobi": _to_float(cmp_row.get("gap_table3_path_minus_gurobi_best_combo_opt")),
                "same_combo_idx": str(cmp_row.get("match_table3_combo_vs_gurobi_best_combo", "")),
                "table3_json": table3_json_path,
                "table3_repaired_json": table3_repaired_json_path,
                "gurobi_json": gurobi_json_path,
            }
        )

        pair_keys = sorted(set(table3_map.keys()) | set(gurobi_map.keys()))
        for s, dep_idx in pair_keys:
            t3_dep = table3_map.get((int(s), int(dep_idx)), {})
            t3_raw_dep = table3_raw_map.get((int(s), int(dep_idx)), {})
            g_dep = gurobi_map.get((int(s), int(dep_idx)), {})
            route_rows.append(
                {
                    "instance": inst,
                    "scenario_index": int(s),
                    "depot_idx_0_4": int(dep_idx),
                    "table3_truck_path_raw_0_4": json.dumps(t3_raw_dep.get("truck_path_0_4", []), ensure_ascii=False),
                    "table3_drone_seq_raw": json.dumps(t3_raw_dep.get("drone_seq", []), ensure_ascii=False),
                    "table3_truck_path_0_4": json.dumps(t3_dep.get("truck_path_0_4", []), ensure_ascii=False),
                    "table3_drone_seq": json.dumps(t3_dep.get("drone_seq", []), ensure_ascii=False),
                    "gurobi_truck_path_0_4": json.dumps(g_dep.get("truck_path_0_4", []), ensure_ascii=False),
                    "gurobi_drone_seq": json.dumps(g_dep.get("drone_seq", []), ensure_ascii=False),
                    "table3_fixed_path_paper_obj": _to_float(cmp_row.get("paper_obj_table3_path_fixed")),
                    "gurobi_paper_obj": _to_float(cmp_row.get("paper_obj_gurobi_best_combo_opt")),
                    "obj_gap_table3_minus_gurobi": _to_float(cmp_row.get("gap_table3_path_minus_gurobi_best_combo_opt")),
                    "table3_runtime_sec": _to_float(unified.get("with_runtime_sec")),
                    "gurobi_runtime_sec": _to_float(cmp_row.get("gurobi_best_runtime_sec")),
                    "same_combo_idx": str(cmp_row.get("match_table3_combo_vs_gurobi_best_combo", "")),
                }
            )

        print(f"[done] {inst}")

    if not instance_rows:
        raise ValueError("No instance rows generated.")

    instance_csv = os.path.join(out_dir, "detailed_instance_comparison.csv")
    instance_json = os.path.join(out_dir, "detailed_instance_comparison.json")
    route_csv = os.path.join(out_dir, "detailed_route_comparison.csv")

    with open(instance_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(instance_rows[0].keys()))
        w.writeheader()
        w.writerows(instance_rows)
    _save_json(instance_json, {"rows": instance_rows, "count": len(instance_rows)})

    with open(route_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(route_rows[0].keys()))
        w.writeheader()
        w.writerows(route_rows)

    print(f"Saved instance CSV: {instance_csv}")
    print(f"Saved instance JSON: {instance_json}")
    print(f"Saved route CSV: {route_csv}")


if __name__ == "__main__":
    main()
