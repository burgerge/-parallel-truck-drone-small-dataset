import argparse
import csv
import json
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Tuple

from gurobi_exact_small_enumeration import _prepare_data


def _to_int(x: Any):
    try:
        return int(x)
    except Exception:
        return None


def _node_to_external(node: Any, dep_to_idx: Dict[int, int]) -> Any:
    if isinstance(node, str) and node.startswith("D") and node[1:].isdigit():
        return node
    iv = _to_int(node)
    if iv is None:
        return node
    if iv in dep_to_idx:
        return f"D{dep_to_idx[iv]}"
    return iv


def _normalize_table3_obj(obj: Dict, dep_to_idx: Dict[int, int], idx_to_dep: Dict[int, int]) -> Dict:
    out = dict(obj)
    raw_x = [int(x) for x in obj.get("x_open", [])]
    if "x_open_internal" in obj:
        x_open_internal = [int(x) for x in obj.get("x_open_internal", [])]
    else:
        if raw_x and all(int(x) in dep_to_idx for x in raw_x):
            x_open_internal = list(raw_x)
        elif raw_x and all(int(x) in idx_to_dep for x in raw_x):
            x_open_internal = [int(idx_to_dep[int(x)]) for x in raw_x]
        else:
            x_open_internal = list(raw_x)
    x_open_idx = [int(dep_to_idx[int(x)]) for x in x_open_internal if int(x) in dep_to_idx]
    out["x_open_internal"] = x_open_internal
    out["x_open"] = x_open_idx
    out["x_open_idx"] = x_open_idx
    out["depot_indexing"] = "candidate_depot_idx_0_to_4"
    out["depot_idx_to_internal"] = {str(k): int(v) for k, v in sorted(idx_to_dep.items())}
    out["depot_internal_to_idx"] = {str(k): int(v) for k, v in sorted(dep_to_idx.items())}

    scenarios = []
    for s in obj.get("scenarios", []):
        rec = dict(s)
        tt_old = s.get("tt", {})
        ds_old = s.get("ds", {})
        tt_new = {}
        ds_new = {}
        for k, walk in tt_old.items():
            ik = _to_int(k)
            if ik is None:
                continue
            if ik in dep_to_idx:
                kk = str(dep_to_idx[ik])
            elif ik in idx_to_dep:
                kk = str(ik)
            else:
                continue
            tt_new[kk] = [_node_to_external(v, dep_to_idx) for v in (walk or [])]
        for k, seq in ds_old.items():
            ik = _to_int(k)
            if ik is None:
                continue
            if ik in dep_to_idx:
                kk = str(dep_to_idx[ik])
            elif ik in idx_to_dep:
                kk = str(ik)
            else:
                continue
            ds_new[kk] = [int(v) for v in (seq or [])]
        rec["tt"] = tt_new
        rec["ds"] = ds_new
        scenarios.append(rec)
    out["scenarios"] = scenarios
    return out


def _normalize_gurobi_obj(obj: Dict, dep_to_idx: Dict[int, int], idx_to_dep: Dict[int, int]) -> Dict:
    out = dict(obj)
    if "open_depots_internal" in obj:
        internal = [int(x) for x in obj.get("open_depots_internal", [])]
    else:
        raw_open = [int(x) for x in obj.get("open_depots", [])]
        if raw_open and all(int(x) in dep_to_idx for x in raw_open):
            internal = list(raw_open)
        elif raw_open and all(int(x) in idx_to_dep for x in raw_open):
            internal = [int(idx_to_dep[int(x)]) for x in raw_open]
        else:
            internal = list(raw_open)
    idx = [int(dep_to_idx[int(x)]) for x in internal if int(x) in dep_to_idx]
    out["open_depots_internal"] = internal
    out["open_depots"] = idx
    out["open_depots_idx"] = idx
    out["depot_indexing"] = "candidate_depot_idx_0_to_4"
    out["depot_idx_to_internal"] = {str(k): int(v) for k, v in sorted(idx_to_dep.items())}
    out["depot_internal_to_idx"] = {str(k): int(v) for k, v in sorted(dep_to_idx.items())}

    res = dict(out.get("result", {}))
    sd = dict(res.get("solution_details", {}))
    scen_rows = []
    for s in sd.get("scenario_routes", []):
        sr = dict(s)
        dep_rows = []
        for d in s.get("depots", []):
            dr = dict(d)
            di = _to_int(d.get("depot_internal"))
            if di is not None and di in dep_to_idx:
                dr["depot_idx"] = int(dep_to_idx[di])
                dr["depot_label"] = f"D{dep_to_idx[di]}"
            tw = d.get("truck_walk", [])
            dr["truck_walk"] = [_node_to_external(v, dep_to_idx) for v in tw]
            tw_e = d.get("truck_walk_euler", [])
            dr["truck_walk_euler"] = [_node_to_external(v, dep_to_idx) for v in tw_e]
            tw_m = d.get("truck_walk_model_order", [])
            dr["truck_walk_model_order"] = [_node_to_external(v, dep_to_idx) for v in tw_m]
            tw_p = d.get("truck_walk_eval_preferred", [])
            dr["truck_walk_eval_preferred"] = [_node_to_external(v, dep_to_idx) for v in tw_p]
            taa = []
            for arc in d.get("truck_active_arcs", []):
                if isinstance(arc, list) and len(arc) == 2:
                    taa.append([_node_to_external(arc[0], dep_to_idx), _node_to_external(arc[1], dep_to_idx)])
            dr["truck_active_arcs"] = taa
            tua = []
            for arc in d.get("truck_unused_arcs_after_walk", []):
                if isinstance(arc, list) and len(arc) == 2:
                    tua.append([_node_to_external(arc[0], dep_to_idx), _node_to_external(arc[1], dep_to_idx)])
            dr["truck_unused_arcs_after_walk"] = tua
            tum = []
            for arc in d.get("truck_unused_arcs_after_model_order", []):
                if isinstance(arc, list) and len(arc) == 2:
                    tum.append([_node_to_external(arc[0], dep_to_idx), _node_to_external(arc[1], dep_to_idx)])
            dr["truck_unused_arcs_after_model_order"] = tum
            twa = []
            for arc in d.get("truck_first_departure_arcs_w", []):
                if isinstance(arc, list) and len(arc) == 2:
                    twa.append([_node_to_external(arc[0], dep_to_idx), _node_to_external(arc[1], dep_to_idx)])
            dr["truck_first_departure_arcs_w"] = twa
            tla = []
            for arc in d.get("truck_last_departure_arcs_a", []):
                if isinstance(arc, list) and len(arc) == 2:
                    tla.append([_node_to_external(arc[0], dep_to_idx), _node_to_external(arc[1], dep_to_idx)])
            dr["truck_last_departure_arcs_a"] = tla
            tbl = []
            for tri in d.get("truck_successor_links_b", []):
                if isinstance(tri, list) and len(tri) == 3:
                    tbl.append(
                        [
                            _node_to_external(tri[0], dep_to_idx),
                            _node_to_external(tri[1], dep_to_idx),
                            _node_to_external(tri[2], dep_to_idx),
                        ]
                    )
            dr["truck_successor_links_b"] = tbl
            dep_rows.append(dr)
        sr["depots"] = dep_rows
        scen_rows.append(sr)
    sd["scenario_routes"] = scen_rows
    res["solution_details"] = sd
    out["result"] = res
    return out


def _dump_json(path: str, obj: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _safe_csv_path(path: str) -> str:
    try:
        with open(path, "w", newline="", encoding="utf-8"):
            pass
        return path
    except PermissionError:
        base, ext = os.path.splitext(path)
        return f"{base}_idx{ext}"


def _safe_json_path(path: str) -> str:
    try:
        with open(path, "w", encoding="utf-8"):
            pass
        return path
    except PermissionError:
        base, ext = os.path.splitext(path)
        return f"{base}_idx{ext}"


def _build_analysis_same_metric(bundle_dir: str) -> None:
    files = [f for f in os.listdir(bundle_dir) if f.endswith(".json")]
    pat_t3 = re.compile(r"^(Instance\d+)_table3_fixedx_combo_([0-9_]+)\.json$")
    pat_gs = re.compile(r"^(Instance\d+)_gurobi_same_as_table3_combo_([0-9_]+)\.json$")
    pat_gb = re.compile(r"^(Instance\d+)_gurobi_best_combo_([0-9_]+)\.json$")
    bucket: Dict[str, Dict[str, str]] = defaultdict(dict)
    for f in files:
        m = pat_t3.match(f)
        if m:
            bucket[m.group(1)]["t3"] = os.path.join(bundle_dir, f)
            continue
        m = pat_gs.match(f)
        if m:
            bucket[m.group(1)]["gs"] = os.path.join(bundle_dir, f)
            continue
        m = pat_gb.match(f)
        if m:
            bucket[m.group(1)]["gb"] = os.path.join(bundle_dir, f)
            continue

    rows = []
    for i in range(1, 11):
        inst = f"Instance{i}"
        rec = bucket.get(inst, {})
        if not all(k in rec for k in ("t3", "gs", "gb")):
            continue
        t3 = json.load(open(rec["t3"], "r", encoding="utf-8"))
        gs = json.load(open(rec["gs"], "r", encoding="utf-8"))
        gb = json.load(open(rec["gb"], "r", encoding="utf-8"))

        t3_combo = sorted([int(x) for x in t3.get("x_open_idx", t3.get("x_open", []))])
        gs_combo = sorted([int(x) for x in gs.get("open_depots_idx", gs.get("open_depots", []))])
        gb_combo = sorted([int(x) for x in gb.get("open_depots_idx", gb.get("open_depots", []))])

        paper_same = gs.get("result", {}).get("objective")
        paper_best = gb.get("result", {}).get("objective")
        status_same = gs.get("result", {}).get("status_name")
        status_best = gb.get("result", {}).get("status_name")
        rt_same = gs.get("result", {}).get("runtime_sec")
        rt_best = gb.get("result", {}).get("runtime_sec")
        arrival_ref = t3.get("expected_obj_a5")

        # path consistency: t3 vs gurobi(same combo)
        t3_scen = {int(s["scenario_index"]): s for s in t3.get("scenarios", [])}
        gs_scen = {
            int(s["scenario_index"]): s
            for s in gs.get("result", {}).get("solution_details", {}).get("scenario_routes", [])
        }
        total_pairs = 0
        truck_exact = 0
        drone_exact = 0
        both_exact = 0
        scen_mis = 0
        for s in sorted(set(t3_scen.keys()) & set(gs_scen.keys())):
            t3_dep = t3_scen[s].get("tt", {})
            t3_ds = t3_scen[s].get("ds", {})
            gs_dep: Dict[int, Dict] = {}
            for d in gs_scen[s].get("depots", []):
                k = _to_int(d.get("depot_idx"))
                if k is None:
                    continue
                gs_dep[k] = d
            scen_all = True
            for k in gs_combo:
                total_pairs += 1
                t3_tr = t3_dep.get(str(k), [])
                gs_tr = gs_dep.get(k, {}).get("truck_walk", [])
                t3_dr = t3_ds.get(str(k), [])
                gs_dr = gs_dep.get(k, {}).get("drone_sequence_spt", [])
                te = (t3_tr == gs_tr)
                de = (t3_dr == gs_dr)
                truck_exact += int(te)
                drone_exact += int(de)
                both_exact += int(te and de)
                if not (te and de):
                    scen_all = False
            if not scen_all:
                scen_mis += 1

        rows.append(
            {
                "instance": inst,
                "table3_combo_idx": t3_combo,
                "gurobi_same_combo_idx": gs_combo,
                "gurobi_best_combo_idx": gb_combo,
                "combo_match_t3_vs_gurobi_best": t3_combo == gb_combo,
                "paper_obj_table3_combo_via_gurobi": paper_same,
                "paper_obj_gurobi_best_combo": paper_best,
                "paper_gap_table3_combo_minus_best": (paper_same - paper_best) if (paper_same is not None and paper_best is not None) else None,
                "paper_status_table3_combo": status_same,
                "paper_status_gurobi_best_combo": status_best,
                "paper_runtime_sec_table3_combo": rt_same,
                "paper_runtime_sec_gurobi_best_combo": rt_best,
                "arrival_obj_table3_reference_only": arrival_ref,
                "path_pairs_total_scenario_depot": total_pairs,
                "truck_walk_exact_ratio_t3_vs_gsame": (truck_exact / total_pairs if total_pairs else None),
                "drone_seq_exact_ratio_t3_vs_gsame": (drone_exact / total_pairs if total_pairs else None),
                "both_exact_ratio_t3_vs_gsame": (both_exact / total_pairs if total_pairs else None),
                "scenario_any_mismatch_count_t3_vs_gsame": scen_mis,
            }
        )

    out_csv = _safe_csv_path(os.path.join(bundle_dir, "analysis_summary_same_metric.csv"))
    out_json = _safe_json_path(os.path.join(bundle_dir, "analysis_summary_same_metric.json"))
    if rows:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize route bundle depot indexing to 0-4 and rebuild same-metric summary."
    )
    parser.add_argument("--bundle-dir", default="outputs/routes_bundle_10instances")
    parser.add_argument("--instances-root", default="./Instance")
    parser.add_argument("--instances", default="1-10")
    args = parser.parse_args()

    if not os.path.isdir(args.bundle_dir):
        raise FileNotFoundError(f"Bundle dir not found: {args.bundle_dir}")

    # Build per-instance depot map
    inst_names: List[str] = []
    raw = (args.instances or "").strip()
    if "-" in raw and "," not in raw:
        a, b = raw.split("-", 1)
        st, ed = int(a), int(b)
        if st > ed:
            st, ed = ed, st
        inst_names = [f"Instance{i}" for i in range(st, ed + 1)]
    else:
        toks = [t.strip() for t in raw.split(",") if t.strip()]
        for t in toks:
            if t.lower().startswith("instance"):
                inst_names.append(t)
            else:
                inst_names.append(f"Instance{int(t)}")
    if not inst_names:
        inst_names = [f"Instance{i}" for i in range(1, 11)]

    maps: Dict[str, Tuple[Dict[int, int], Dict[int, int]]] = {}
    for inst in inst_names:
        _, _, _, candidates = _prepare_data(inst, args.instances_root, 0, 0)
        dep_to_idx = {int(k): int(i) for i, k in enumerate(candidates)}
        idx_to_dep = {int(i): int(k) for i, k in enumerate(candidates)}
        maps[inst] = (dep_to_idx, idx_to_dep)

    for fn in os.listdir(args.bundle_dir):
        if not fn.endswith(".json"):
            continue
        full = os.path.join(args.bundle_dir, fn)
        m = re.match(r"^(Instance\d+)_", fn)
        if not m:
            continue
        inst = m.group(1)
        if inst not in maps:
            continue
        dep_to_idx, idx_to_dep = maps[inst]
        obj = json.load(open(full, "r", encoding="utf-8"))
        if fn.find("_table3_fixedx_combo_") >= 0:
            out = _normalize_table3_obj(obj, dep_to_idx, idx_to_dep)
            _dump_json(full, out)
        elif fn.find("_gurobi_") >= 0:
            out = _normalize_gurobi_obj(obj, dep_to_idx, idx_to_dep)
            _dump_json(full, out)

    # Rebuild route_bundle_index in idx form
    rows = []
    pat_t3 = re.compile(r"^(Instance\d+)_table3_fixedx_combo_([0-9_]+)\.json$")
    pat_gs = re.compile(r"^(Instance\d+)_gurobi_same_as_table3_combo_([0-9_]+)\.json$")
    pat_gb = re.compile(r"^(Instance\d+)_gurobi_best_combo_([0-9_]+)\.json$")
    by_inst: Dict[str, Dict[str, str]] = defaultdict(dict)
    for fn in os.listdir(args.bundle_dir):
        m = pat_t3.match(fn)
        if m:
            by_inst[m.group(1)]["t3"] = os.path.join(args.bundle_dir, fn)
            continue
        m = pat_gs.match(fn)
        if m:
            by_inst[m.group(1)]["gs"] = os.path.join(args.bundle_dir, fn)
            continue
        m = pat_gb.match(fn)
        if m:
            by_inst[m.group(1)]["gb"] = os.path.join(args.bundle_dir, fn)
            continue

    for inst in sorted(by_inst.keys(), key=lambda x: int(x.replace("Instance", ""))):
        rec = by_inst[inst]
        if "t3" not in rec or "gb" not in rec:
            continue
        t3 = json.load(open(rec["t3"], "r", encoding="utf-8"))
        gb = json.load(open(rec["gb"], "r", encoding="utf-8"))
        gs_combo_idx = []
        if "gs" in rec:
            gs = json.load(open(rec["gs"], "r", encoding="utf-8"))
            gs_combo_idx = gs.get("open_depots_idx", gs.get("open_depots", []))
        rows.append(
            {
                "instance": inst,
                "table3_combo_idx": t3.get("x_open_idx", t3.get("x_open", [])),
                "gurobi_best_combo_idx": gb.get("open_depots_idx", gb.get("open_depots", [])),
                "gurobi_same_as_table3_combo_idx": gs_combo_idx,
                "table3_paths_json": rec.get("t3", ""),
                "gurobi_best_paths_json": rec.get("gb", ""),
                "gurobi_same_as_table3_paths_json": rec.get("gs", ""),
            }
        )

    idx_json = _safe_json_path(os.path.join(args.bundle_dir, "route_bundle_index.json"))
    idx_csv = _safe_csv_path(os.path.join(args.bundle_dir, "route_bundle_index.csv"))
    with open(idx_json, "w", encoding="utf-8") as f:
        json.dump({"rows": rows, "count": len(rows)}, f, ensure_ascii=False, indent=2)
    with open(idx_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "instance",
                "table3_combo_idx",
                "gurobi_best_combo_idx",
                "gurobi_same_as_table3_combo_idx",
                "table3_paths_json",
                "gurobi_best_paths_json",
                "gurobi_same_as_table3_paths_json",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r["instance"],
                    r["table3_combo_idx"],
                    r["gurobi_best_combo_idx"],
                    r["gurobi_same_as_table3_combo_idx"],
                    r["table3_paths_json"],
                    r["gurobi_best_paths_json"],
                    r["gurobi_same_as_table3_paths_json"],
                ]
            )

    _build_analysis_same_metric(args.bundle_dir)
    print(f"Normalized bundle to idx: {args.bundle_dir}")


if __name__ == "__main__":
    main()
