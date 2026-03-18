import argparse
import csv
import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

from compare_unified_scoring import (
    _evaluate_by_arrival_sim,
    _extract_gurobi_paths,
    _extract_table3_paths,
    _first_visit_demands,
)
from gurobi_exact_small_enumeration import _prepare_data
from paper_eval_common import parse_instance_names


def _collect_bundle(bundle_dir: str) -> Dict[str, Dict[str, str]]:
    by_inst: Dict[str, Dict[str, str]] = defaultdict(dict)
    for fn in os.listdir(bundle_dir):
        if not fn.endswith(".json"):
            continue
        full = os.path.join(bundle_dir, fn)
        m = re.match(r"^(Instance\d+)_table3_fixedx_combo_.*\.json$", fn)
        if m:
            by_inst[m.group(1)]["t3"] = full
            continue
        m = re.match(r"^(Instance\d+)_gurobi_same_as_table3_combo_.*\.json$", fn)
        if m:
            by_inst[m.group(1)]["gs"] = full
            continue
        m = re.match(r"^(Instance\d+)_gurobi_best_combo_.*\.json$", fn)
        if m:
            by_inst[m.group(1)]["gb"] = full
            continue
    return by_inst


def _table3_open_internal(payload: Dict) -> List[int]:
    if "x_open_internal" in payload:
        return sorted(int(x) for x in payload.get("x_open_internal", []))

    idx_to_internal_raw = payload.get("depot_idx_to_internal", {})
    idx_to_internal = {int(k): int(v) for k, v in idx_to_internal_raw.items()}
    vals = []
    for x in payload.get("x_open", []):
        xi = int(x)
        vals.append(int(idx_to_internal.get(xi, xi)))
    return sorted(vals)


def _analysis_by_instance(path: str) -> Dict[str, Dict]:
    if not path or not os.path.exists(path):
        return {}
    rows = list(csv.DictReader(open(path, "r", encoding="utf-8")))
    return {str(r.get("instance", "")).strip(): dict(r) for r in rows}


def _to_float(row: Dict, key: str):
    raw = row.get(key)
    if raw in (None, ""):
        return None
    return float(raw)


def _pick_best_combo_idx(gb_payload: Dict, analysis_row: Dict) -> List[int]:
    if analysis_row:
        raw = str(analysis_row.get("gurobi_best_combo_idx", "")).strip()
        if raw:
            try:
                return [int(x) for x in json.loads(raw)]
            except Exception:
                pass
    vals = gb_payload.get("open_depots_idx", gb_payload.get("open_depots", []))
    return [int(x) for x in vals]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-validate explicit route similarity between Table3 fixed-x and Gurobi same-combo outputs."
    )
    parser.add_argument("--bundle-dir", default="outputs/routes_bundle_10instances")
    parser.add_argument("--instances-root", default="./Instance")
    parser.add_argument("--instances", default="1-10")
    parser.add_argument("--analysis-csv", default="")
    parser.add_argument("--out-summary-csv", default="")
    parser.add_argument("--out-summary-json", default="")
    parser.add_argument("--out-detail-csv", default="")
    args = parser.parse_args()

    bundle = _collect_bundle(args.bundle_dir)
    analysis_rows = _analysis_by_instance(args.analysis_csv)

    summary_rows: List[Dict] = []
    detail_rows: List[Dict] = []

    for inst in parse_instance_names(args.instances):
        rec = bundle.get(inst, {})
        if not all(k in rec for k in ("t3", "gs", "gb")):
            continue

        t3_payload = json.load(open(rec["t3"], "r", encoding="utf-8"))
        gs_payload = json.load(open(rec["gs"], "r", encoding="utf-8"))
        gb_payload = json.load(open(rec["gb"], "r", encoding="utf-8"))

        t3_tt_by_s, t3_ds_by_s, weights = _extract_table3_paths(t3_payload)
        gs_tt_by_s, gs_ds_by_s = _extract_gurobi_paths(gs_payload)
        gb_tt_by_s, gb_ds_by_s = _extract_gurobi_paths(gb_payload)
        open_internal = _table3_open_internal(t3_payload)

        _base, scenarios, demand_nodes, _candidates = _prepare_data(inst, args.instances_root, 0, 0)
        demand_set = set(int(i) for i in demand_nodes)

        arr_t3 = _evaluate_by_arrival_sim(scenarios, t3_tt_by_s, t3_ds_by_s, weights)
        arr_gs = _evaluate_by_arrival_sim(scenarios, gs_tt_by_s, gs_ds_by_s, weights)
        arr_gb = _evaluate_by_arrival_sim(scenarios, gb_tt_by_s, gb_ds_by_s, weights)

        exact_walk_count = 0
        first_visit_count = 0
        drone_seq_count = 0
        pair_count = 0

        n_scen = min(len(scenarios), len(t3_tt_by_s), len(gs_tt_by_s))
        for s in range(n_scen):
            for k in open_internal:
                t3_walk = [int(x) for x in t3_tt_by_s[s].get(int(k), [])]
                gs_walk = [int(x) for x in gs_tt_by_s[s].get(int(k), [])]
                t3_ds = [int(x) for x in t3_ds_by_s[s].get(int(k), [])]
                gs_ds = [int(x) for x in gs_ds_by_s[s].get(int(k), [])]

                same_walk = int(t3_walk == gs_walk)
                same_first_visit = int(
                    _first_visit_demands(t3_walk, demand_set) == _first_visit_demands(gs_walk, demand_set)
                )
                same_drone = int(t3_ds == gs_ds)

                exact_walk_count += same_walk
                first_visit_count += same_first_visit
                drone_seq_count += same_drone
                pair_count += 1

                detail_rows.append(
                    {
                        "instance": inst,
                        "scenario_index": int(s),
                        "depot_internal": int(k),
                        "same_walk_exact": same_walk,
                        "same_first_visit_order": same_first_visit,
                        "same_drone_seq_exact": same_drone,
                    }
                )

        analysis_row = analysis_rows.get(inst, {})
        paper_t3 = _to_float(analysis_row, "paper_obj_table3_path_fixed")
        paper_same = _to_float(analysis_row, "paper_obj_gurobi_same_combo_opt")
        paper_best = _to_float(analysis_row, "paper_obj_gurobi_best_combo_opt")
        paper_gap_same = _to_float(analysis_row, "gap_table3_path_minus_gurobi_same_combo_opt")
        paper_gap_best = _to_float(analysis_row, "gap_table3_path_minus_gurobi_best_combo_opt")

        summary_rows.append(
            {
                "instance": inst,
                "table3_combo_internal": open_internal,
                "gurobi_best_combo_idx": _pick_best_combo_idx(gb_payload, analysis_row),
                "paper_obj_table3_path_fixed": paper_t3,
                "paper_obj_gurobi_same_combo_opt": paper_same,
                "paper_obj_gurobi_best_combo_opt": paper_best,
                "paper_gap_t3_minus_same": paper_gap_same,
                "paper_gap_t3_minus_best": paper_gap_best,
                "arrival_obj_table3": arr_t3.get("weighted_avg"),
                "arrival_obj_gurobi_same": arr_gs.get("weighted_avg"),
                "arrival_obj_gurobi_best": arr_gb.get("weighted_avg"),
                "arrival_gap_t3_minus_same": (
                    None
                    if arr_t3.get("weighted_avg") is None or arr_gs.get("weighted_avg") is None
                    else float(arr_t3.get("weighted_avg")) - float(arr_gs.get("weighted_avg"))
                ),
                "arrival_gap_t3_minus_best": (
                    None
                    if arr_t3.get("weighted_avg") is None or arr_gb.get("weighted_avg") is None
                    else float(arr_t3.get("weighted_avg")) - float(arr_gb.get("weighted_avg"))
                ),
                "same_combo_pair_count": int(pair_count),
                "same_combo_exact_walk_match_count": int(exact_walk_count),
                "same_combo_first_visit_match_count": int(first_visit_count),
                "same_combo_exact_drone_seq_match_count": int(drone_seq_count),
            }
        )

        print(f"[done] {inst}")

    if not summary_rows:
        raise ValueError("No complete instances found for cross validation.")

    out_summary_csv = args.out_summary_csv or os.path.join(args.bundle_dir, "cross_validation_summary.csv")
    out_summary_json = args.out_summary_json or os.path.join(args.bundle_dir, "cross_validation_summary.json")
    out_detail_csv = args.out_detail_csv or os.path.join(args.bundle_dir, "cross_validation_path_match_details.csv")

    with open(out_summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    with open(out_summary_json, "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, ensure_ascii=False, indent=2)
    with open(out_detail_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(detail_rows[0].keys()))
        w.writeheader()
        w.writerows(detail_rows)

    print(f"Saved summary CSV: {out_summary_csv}")
    print(f"Saved summary JSON: {out_summary_json}")
    print(f"Saved detail CSV: {out_detail_csv}")


if __name__ == "__main__":
    main()
