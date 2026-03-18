import argparse
import csv
import json
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from compare_unified_scoring import _build_strict_fixed_from_table3_paths
from gurobi_exact_small_enumeration import _prepare_data, solve_fixed_combo_commodity


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
        return int(idx_to_internal.get(int(idx), idx))
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


def _extract_table3_paths_internal(t3: Dict) -> Tuple[List[Dict[int, List[int]]], List[Dict[int, List[int]]], List[int]]:
    if "x_open_internal" in t3:
        open_internal = [int(x) for x in t3.get("x_open_internal", [])]
    else:
        open_internal = [int(x) for x in t3.get("x_open", [])]
    internal_set = set(open_internal)

    idx_to_internal_raw = t3.get("depot_idx_to_internal", {})
    idx_to_internal = {int(k): int(v) for k, v in idx_to_internal_raw.items()}

    tt_by_s: List[Dict[int, List[int]]] = []
    ds_by_s: List[Dict[int, List[int]]] = []
    for s in t3.get("scenarios", []):
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

    return tt_by_s, ds_by_s, sorted(open_internal)


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Table3 routes under paper_exact by fixing binaries, then compare to Gurobi objectives."
    )
    parser.add_argument("--bundle-dir", default="outputs/routes_bundle_10instances")
    parser.add_argument("--instances-root", default="./Instance")
    parser.add_argument("--instances", default="1-10")
    parser.add_argument("--time-limit", type=float, default=600.0)
    parser.add_argument("--mip-gap", type=float, default=0.0)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--mip-focus", type=int, default=1)
    parser.add_argument("--presolve", type=int, default=2)
    parser.add_argument("--cuts", type=int, default=2)
    parser.add_argument("--symmetry", type=int, default=2)
    parser.add_argument("--out-csv", default="")
    parser.add_argument("--out-json", default="")
    args = parser.parse_args()

    by_inst = _collect_bundle(args.bundle_dir)
    out_rows: List[Dict] = []

    raw = (args.instances or "").strip()
    inst_names: List[str] = []
    if "-" in raw and "," not in raw:
        a, b = raw.split("-", 1)
        st, ed = int(a), int(b)
        if st > ed:
            st, ed = ed, st
        inst_names = [f"Instance{i}" for i in range(st, ed + 1)]
    else:
        for t in [x.strip() for x in raw.split(",") if x.strip()]:
            if t.lower().startswith("instance"):
                inst_names.append(t)
            else:
                inst_names.append(f"Instance{int(t)}")
    if not inst_names:
        inst_names = [f"Instance{i}" for i in range(1, 11)]

    for inst in inst_names:
        rec = by_inst.get(inst, {})
        if not all(k in rec for k in ("t3", "gs", "gb")):
            continue

        t3 = json.load(open(rec["t3"], "r", encoding="utf-8"))
        gs = json.load(open(rec["gs"], "r", encoding="utf-8"))
        gb = json.load(open(rec["gb"], "r", encoding="utf-8"))

        tt_by_s, ds_by_s, open_internal = _extract_table3_paths_internal(t3)
        _, scenarios, demand_nodes, candidates = _prepare_data(inst, args.instances_root, 0, 0)
        pos = {int(k): i for i, k in enumerate(candidates)}

        fixed_t3 = _build_strict_fixed_from_table3_paths(
            scenarios=scenarios,
            demand_nodes=demand_nodes,
            open_depots=open_internal,
            tt_by_s=tt_by_s,
            ds_by_s=ds_by_s,
        )

        eval_t3_path = solve_fixed_combo_commodity(
            instance_name=inst,
            scenarios=scenarios,
            demand_nodes=demand_nodes,
            open_depots=open_internal,
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

        paper_t3_path_obj = eval_t3_path.get("objective")
        paper_t3_path_status = eval_t3_path.get("status_name")
        paper_same_combo_opt = gs.get("result", {}).get("objective")
        paper_best_combo_opt = gb.get("result", {}).get("objective")
        gap_vs_same = None
        if paper_t3_path_obj is not None and paper_same_combo_opt is not None:
            gap_vs_same = float(paper_t3_path_obj) - float(paper_same_combo_opt)
        gap_vs_best = None
        if paper_t3_path_obj is not None and paper_best_combo_opt is not None:
            gap_vs_best = float(paper_t3_path_obj) - float(paper_best_combo_opt)

        out_rows.append(
            {
                "instance": inst,
                "table3_combo_internal": list(open_internal),
                "table3_combo_idx": sorted(pos[int(k)] for k in open_internal if int(k) in pos),
                "gurobi_best_combo_idx": gb.get("open_depots_idx", gb.get("open_depots", [])),
                "paper_obj_table3_path_fixed": paper_t3_path_obj,
                "paper_status_table3_path_fixed": paper_t3_path_status,
                "paper_obj_gurobi_same_combo_opt": paper_same_combo_opt,
                "paper_obj_gurobi_best_combo_opt": paper_best_combo_opt,
                "gap_table3_path_minus_gurobi_same_combo_opt": gap_vs_same,
                "gap_table3_path_minus_gurobi_best_combo_opt": gap_vs_best,
                "arrival_obj_table3_reference": t3.get("expected_obj_a5"),
                "paper_fixing_mode": "strict_y_c_z_w_a_b",
            }
        )
        print(f"[done] {inst}")

    if not out_rows:
        raise ValueError("No complete instances found in bundle dir.")

    out_csv = args.out_csv or os.path.join(args.bundle_dir, "analysis_table3_path_fixed_vs_gurobi.csv")
    out_json = args.out_json or os.path.join(args.bundle_dir, "analysis_table3_path_fixed_vs_gurobi.json")

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out_rows, f, ensure_ascii=False, indent=2)

    print(f"Saved CSV: {out_csv}")
    print(f"Saved JSON: {out_json}")


if __name__ == "__main__":
    main()
