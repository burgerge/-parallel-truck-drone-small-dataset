import argparse
import json
import math
import os
from collections import Counter
from typing import Dict, List, Optional, Sequence, Set, Tuple

import alg
from compare_unified_scoring import (
    _build_fixed_from_table3_paths,
    _close_tour_by_shortest_path,
    _extract_gurobi_paths,
    _extract_table3_paths,
)
from gurobi_exact_small_enumeration import _prepare_data, solve_fixed_combo_commodity


def _load_open_depots_from_table3(payload: Dict) -> List[int]:
    if "x_open_internal" in payload:
        return sorted(int(x) for x in payload.get("x_open_internal", []))

    idx_to_internal_raw = payload.get("depot_idx_to_internal", {})
    idx_to_internal = {int(k): int(v) for k, v in idx_to_internal_raw.items()}
    vals = []
    for x in payload.get("x_open", []):
        xi = int(x)
        vals.append(int(idx_to_internal.get(xi, xi)))
    return sorted(vals)


def _load_open_depots_from_gurobi(payload: Dict) -> List[int]:
    vals = payload.get("open_depots_internal", [])
    return sorted(int(x) for x in vals)


def _normalize_explicit_source(
    table3_json: Optional[str],
    gurobi_json: Optional[str],
) -> Tuple[str, Dict, List[int], List[Dict[int, List[int]]], List[Dict[int, List[int]]]]:
    if bool(table3_json) == bool(gurobi_json):
        raise ValueError("Provide exactly one of --table3-json or --gurobi-json.")

    if table3_json:
        payload = json.load(open(table3_json, "r", encoding="utf-8"))
        open_depots = _load_open_depots_from_table3(payload)
        tt_by_s, ds_by_s, _weights = _extract_table3_paths(payload)
        return "table3", payload, open_depots, tt_by_s, ds_by_s

    payload = json.load(open(gurobi_json, "r", encoding="utf-8"))
    open_depots = _load_open_depots_from_gurobi(payload)
    tt_by_s, ds_by_s = _extract_gurobi_paths(payload)
    return "gurobi", payload, open_depots, tt_by_s, ds_by_s


def _first_visit_demands(tour: Sequence[int], demand_set: set) -> List[int]:
    seen = set()
    out: List[int] = []
    for node in tour:
        node = int(node)
        if node not in demand_set or node in seen:
            continue
        seen.add(node)
        out.append(node)
    return out


def _compress_consecutive_duplicates(tour: Sequence[int]) -> List[int]:
    out: List[int] = []
    for node in tour or []:
        node = int(node)
        if out and int(out[-1]) == int(node):
            continue
        out.append(int(node))
    return out


def _group_fixed_assignments(
    fixed_binary_values: Dict[str, set],
    open_depots: Sequence[int],
) -> Dict[str, Dict[int, List]]:
    y_by_depot: Dict[int, List[List[int]]] = {int(k): [] for k in open_depots}
    c_by_depot: Dict[int, List[int]] = {int(k): [] for k in open_depots}
    z_by_depot: Dict[int, List[int]] = {int(k): [] for k in open_depots}
    w_by_depot: Dict[int, List[List[int]]] = {int(k): [] for k in open_depots}
    a_by_depot: Dict[int, List[List[int]]] = {int(k): [] for k in open_depots}
    b_links: List[List[int]] = []

    for (_s, k, a, b) in sorted(fixed_binary_values.get("y_on", set())):
        y_by_depot[int(k)].append([int(a), int(b)])
    for (_s, k, i) in sorted(fixed_binary_values.get("c_on", set())):
        c_by_depot[int(k)].append(int(i))
    for (_s, k, i) in sorted(fixed_binary_values.get("z_on", set())):
        z_by_depot[int(k)].append(int(i))
    for (_s, k, i, j) in sorted(fixed_binary_values.get("w_on", set())):
        w_by_depot[int(k)].append([int(i), int(j)])
    for (_s, k, i, j) in sorted(fixed_binary_values.get("a_on", set())):
        a_by_depot[int(k)].append([int(i), int(j)])
    for (_s, i, j, l) in sorted(fixed_binary_values.get("b_on", set())):
        b_links.append([int(i), int(j), int(l)])

    return {
        "y_on_by_depot": y_by_depot,
        "c_on_by_depot": c_by_depot,
        "z_on_by_depot": z_by_depot,
        "w_on_by_depot": w_by_depot,
        "a_on_by_depot": a_by_depot,
        "b_on_links": b_links,
    }


def _summarize_explicit_paths(
    scen,
    demand_nodes: Sequence[int],
    open_depots: Sequence[int],
    tt: Dict[int, List[int]],
    ds: Dict[int, List[int]],
) -> Dict:
    demand_set = set(int(i) for i in demand_nodes)
    recs = []

    for k in sorted(int(x) for x in open_depots):
        raw_tour = [int(x) for x in tt.get(int(k), [int(k)])]
        normalized_start = (raw_tour[:1] != [int(k)])
        closed_tour = _close_tour_by_shortest_path(scen, raw_tour, int(k), demand_nodes)
        auto_closed = bool(closed_tour) and int(closed_tour[-1]) == int(k) and (
            not raw_tour or int(raw_tour[-1]) != int(k)
        )

        raw_arcs = [(int(a), int(b)) for (a, b) in zip(closed_tour[:-1], closed_tour[1:]) if int(a) != int(b)]
        arc_counts = Counter(raw_arcs)
        repeated_arcs = [
            {"arc": [int(a), int(b)], "count": int(cnt)}
            for (a, b), cnt in sorted(arc_counts.items())
            if int(cnt) > 1
        ]
        recs.append(
            {
                "depot_internal": int(k),
                "raw_tour": raw_tour,
                "closed_tour_used_for_y": [int(x) for x in closed_tour],
                "normalized_start_inserted": bool(normalized_start),
                "auto_closed_back_to_depot": bool(auto_closed),
                "raw_arc_count": len(raw_arcs),
                "unique_arc_count": len(arc_counts),
                "repeated_directed_arcs": repeated_arcs,
                "first_visit_demands_in_tour": _first_visit_demands(closed_tour[1:], demand_set),
                "drone_sequence": [int(x) for x in ds.get(int(k), [])],
            }
        )
    return {"depots": recs}


def _build_order_fixed_binaries_from_explicit_paths(
    scen,
    demand_nodes: Sequence[int],
    open_depots: Sequence[int],
    tt: Dict[int, List[int]],
    ds: Dict[int, List[int]],
) -> Tuple[Dict[str, Set[Tuple[int, ...]]], Dict]:
    demand_set = set(int(i) for i in demand_nodes)
    fixed: Dict[str, Set[Tuple[int, ...]]] = {
        "y_on": set(),
        "c_on": set(),
        "z_on": set(),
        "w_on": set(),
        "a_on": set(),
        "b_on": set(),
    }
    meta = {"depots": [], "warnings": []}

    for k in sorted(int(x) for x in open_depots):
        raw_tour = [int(x) for x in tt.get(int(k), [int(k)])]
        norm_tour = _compress_consecutive_duplicates(raw_tour)
        if not norm_tour or int(norm_tour[0]) != int(k):
            norm_tour = [int(k)] + [int(x) for x in norm_tour if int(x) != int(k) or not norm_tour]
            norm_tour = _compress_consecutive_duplicates(norm_tour)
        closed_tour = _close_tour_by_shortest_path(scen, norm_tour, int(k), demand_nodes)
        closed_tour = _compress_consecutive_duplicates(closed_tour)

        route_nodes = demand_set | {int(k)}
        invalid_arcs: List[List[int]] = []
        kept_arcs: List[List[int]] = []
        for a, b in zip(closed_tour[:-1], closed_tour[1:]):
            a = int(a)
            b = int(b)
            if a == b:
                continue
            if a not in route_nodes or b not in route_nodes:
                invalid_arcs.append([int(a), int(b)])
                continue
            if math.isinf(float(scen.truck_times.get(int(a), {}).get(int(b), math.inf))):
                invalid_arcs.append([int(a), int(b)])
                continue
            fixed["y_on"].add((0, int(k), int(a), int(b)))
            kept_arcs.append([int(a), int(b)])

        drone_seq = [int(x) for x in ds.get(int(k), [])]
        for i in drone_seq:
            if int(i) not in demand_set:
                meta["warnings"].append(f"Depot {k}: drone node {i} is not a demand node and was skipped.")
                continue
            fixed["z_on"].add((0, int(k), int(i)))

        truck_demands = _first_visit_demands(closed_tour[1:], demand_set)
        for i in truck_demands:
            fixed["c_on"].add((0, int(k), int(i)))

        occurrences: Dict[int, List[Tuple[int, int]]] = {}
        for pos in range(1, len(closed_tour) - 1):
            prev_node = int(closed_tour[pos - 1])
            cur_node = int(closed_tour[pos])
            next_node = int(closed_tour[pos + 1])
            if int(cur_node) not in demand_set:
                continue
            if (0, int(k), int(cur_node), int(next_node)) not in fixed["y_on"]:
                meta["warnings"].append(
                    f"Depot {k}: skipped order hint at node {cur_node} because arc ({cur_node},{next_node}) is not in fixed y."
                )
                continue
            occurrences.setdefault(int(cur_node), []).append((int(prev_node), int(next_node)))

        for i, occs in sorted(occurrences.items()):
            first_pred, first_next = occs[0]
            fixed["w_on"].add((0, int(k), int(i), int(first_next)))

            last_pred, last_next = occs[-1]
            fixed["a_on"].add((0, int(k), int(i), int(last_next)))

            for pred, nxt in occs[1:]:
                if (0, int(k), int(pred), int(i)) not in fixed["y_on"]:
                    meta["warnings"].append(
                        f"Depot {k}: skipped b hint ({pred},{i},{nxt}) because predecessor arc is not in fixed y."
                    )
                    continue
                if (0, int(k), int(i), int(nxt)) not in fixed["y_on"]:
                    meta["warnings"].append(
                        f"Depot {k}: skipped b hint ({pred},{i},{nxt}) because successor arc is not in fixed y."
                    )
                    continue
                fixed["b_on"].add((0, int(pred), int(i), int(nxt)))

        meta["depots"].append(
            {
                "depot_internal": int(k),
                "raw_tour": raw_tour,
                "normalized_tour": norm_tour,
                "closed_tour_used_for_model": [int(x) for x in closed_tour],
                "removed_consecutive_duplicates": max(0, len(raw_tour) - len(norm_tour)),
                "auto_closed_back_to_depot": bool(closed_tour) and int(closed_tour[-1]) == int(k) and (
                    not norm_tour or int(norm_tour[-1]) != int(k)
                ),
                "kept_arcs": kept_arcs,
                "invalid_or_skipped_arcs": invalid_arcs,
                "truck_demands_fixed_in_c": truck_demands,
                "drone_demands_fixed_in_z": [int(x) for x in drone_seq if int(x) in demand_set],
            }
        )

    return fixed, meta


def _extract_reconstructed_solution(solution_details: Optional[Dict]) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    tt: Dict[int, List[int]] = {}
    ds: Dict[int, List[int]] = {}
    if not solution_details:
        return tt, ds

    rows = solution_details.get("scenario_routes", [])
    if not rows:
        return tt, ds

    for dep in rows[0].get("depots", []):
        k = int(dep.get("depot_internal"))
        walk = dep.get("truck_walk_eval_preferred", []) or dep.get("truck_walk_model_order", []) or dep.get("truck_walk", [])
        tt[int(k)] = [int(x) for x in walk]
        ds[int(k)] = [int(x) for x in dep.get("drone_sequence_spt", [])]
    return tt, ds


def _index_debug_values(rows: Sequence[Sequence[float]], key_len: int) -> Dict[Tuple[int, ...], float]:
    out: Dict[Tuple[int, ...], float] = {}
    for row in rows or []:
        key = tuple(int(row[idx]) for idx in range(key_len))
        out[key] = float(row[key_len])
    return out


def _expected_flow_from_walk(
    strict_meta: Dict,
    strict_fixed_binary_values: Dict[str, Set[Tuple[int, ...]]],
) -> Dict[str, object]:
    c_on = {
        (int(s), int(k), int(i))
        for (s, k, i) in strict_fixed_binary_values.get("c_on", set())
    }
    y_on = {
        (int(s), int(k), int(a), int(b))
        for (s, k, a, b) in strict_fixed_binary_values.get("y_on", set())
    }

    f_expected: Dict[Tuple[int, int, int], float] = {}
    sink_expected: Dict[Tuple[int, int], float] = {}
    per_depot = []

    for dep in strict_meta.get("depots", []):
        k = int(dep.get("depot_internal"))
        walk = [int(x) for x in dep.get("closed_tour_used_for_model", [])]
        truck_demands = set(int(x) for x in dep.get("truck_demands_fixed_in_c", []))
        remaining = float(len(truck_demands) + 1)
        first_departed: Set[int] = set()
        arc_recs = []
        last_non_depot = None

        for a, b in zip(walk[:-1], walk[1:]):
            a = int(a)
            b = int(b)
            if a in truck_demands and a not in first_departed:
                remaining -= 1.0
                first_departed.add(int(a))
            if (0, int(k), int(a), int(b)) not in y_on:
                arc_recs.append({"arc": [int(a), int(b)], "expected_flow": None, "note": "arc_not_in_fixed_y"})
                continue
            f_expected[(0, int(a), int(b))] = float(remaining)
            arc_recs.append({"arc": [int(a), int(b)], "expected_flow": float(remaining)})
            if b == k:
                last_non_depot = int(a)

        if last_non_depot is not None:
            sink_expected[(0, int(last_non_depot))] = 1.0

        per_depot.append(
            {
                "depot_internal": int(k),
                "closed_tour_used_for_model": walk,
                "truck_demands_fixed_in_c": sorted(int(x) for x in truck_demands),
                "expected_arc_flows": arc_recs,
                "expected_sink_from_node": last_non_depot,
            }
        )

    return {
        "f_expected_on_arcs": [
            [int(s), int(i), int(j), float(v)]
            for (s, i, j), v in sorted(f_expected.items())
        ],
        "sink_expected": [
            [int(s), int(i), float(v)]
            for (s, i), v in sorted(sink_expected.items())
        ],
        "per_depot": per_depot,
    }


def _compare_expected_vs_model_flow(
    expected: Dict[str, object],
    debug_dump: Optional[Dict],
) -> Dict[str, object]:
    if not debug_dump:
        return {"available": False}

    model_f = _index_debug_values(debug_dump.get("f_positive", []), key_len=3)
    expected_f = _index_debug_values(expected.get("f_expected_on_arcs", []), key_len=3)
    expected_sink = _index_debug_values(expected.get("sink_expected", []), key_len=2)

    arc_rows = []
    mismatches = []
    for key in sorted(set(expected_f.keys()) | set(model_f.keys())):
        exp = expected_f.get(key)
        act = model_f.get(key)
        diff = None if exp is None or act is None else float(act) - float(exp)
        row = [int(key[0]), int(key[1]), int(key[2]), exp, act, diff]
        arc_rows.append(row)
        if exp is None or act is None or abs(float(diff)) > 1e-9:
            mismatches.append(row)

    sink_rows = []
    for key in sorted(expected_sink.keys()):
        act = model_f.get((int(key[0]), int(key[1]), -1))
        diff = None if act is None else float(act) - float(expected_sink[key])
        sink_rows.append([int(key[0]), int(key[1]), expected_sink[key], act, diff])

    return {
        "available": True,
        "arc_flow_rows": arc_rows,
        "arc_flow_mismatches": mismatches,
        "sink_rows": sink_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Debug one explicit TTk/DSk solution by scoring it with "
            "alg.calculate_arrival_times and with paper_exact under fixed binaries."
        )
    )
    parser.add_argument("--instance", required=True)
    parser.add_argument("--instances-root", default="./Instance")
    parser.add_argument("--table3-json", default="")
    parser.add_argument("--gurobi-json", default="")
    parser.add_argument("--scenario-index", type=int, default=0)
    parser.add_argument("--time-limit", type=float, default=120.0)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    source_kind, source_payload, open_depots, tt_by_s, ds_by_s = _normalize_explicit_source(
        table3_json=(args.table3_json or "").strip(),
        gurobi_json=(args.gurobi_json or "").strip(),
    )

    if not open_depots:
        raise ValueError("Could not determine open depots from input JSON.")

    _base, scenarios, demand_nodes, _candidates = _prepare_data(args.instance, args.instances_root, 0, 0)
    s = int(args.scenario_index)
    if s < 0 or s >= len(scenarios):
        raise ValueError(f"--scenario-index out of range: {s}, available={len(scenarios)}")
    if s >= len(tt_by_s) or s >= len(ds_by_s):
        raise ValueError(f"Input JSON does not contain scenario {s}.")

    scen = scenarios[s]
    tt = {int(k): [int(x) for x in v] for k, v in tt_by_s[s].items()}
    ds = {int(k): [int(x) for x in v] for k, v in ds_by_s[s].items()}

    cfg = alg.HeuristicConfig(
        num_depots_to_open=max(1, len(open_depots)),
        num_scenarios=1,
        seed=123,
        k_max=7,
        l_max=6,
        i_max=5,
        drone_time_is_roundtrip=True,
        normalize_by_num_demands=True,
        strict_feasibility=True,
    )
    arrival_obj = float(alg.calculate_arrival_times(tt, ds, scen, cfg))
    strict_fixed_binary_values, strict_meta = _build_order_fixed_binaries_from_explicit_paths(
        scen=scen,
        demand_nodes=demand_nodes,
        open_depots=open_depots,
        tt=tt,
        ds=ds,
    )
    strict_tt = {
        int(item["depot_internal"]): [int(x) for x in item.get("closed_tour_used_for_model", [])]
        for item in strict_meta.get("depots", [])
    }
    strict_arrival_obj = float(alg.calculate_arrival_times(strict_tt, ds, scen, cfg))

    fixed_binary_values = _build_fixed_from_table3_paths(
        scenarios=[scen],
        demand_nodes=demand_nodes,
        open_depots=open_depots,
        tt_by_s=[tt],
        ds_by_s=[ds],
    )

    paper_res_relaxed = solve_fixed_combo_commodity(
        instance_name=args.instance,
        scenarios=[scen],
        demand_nodes=demand_nodes,
        open_depots=open_depots,
        time_limit=float(args.time_limit),
        mip_gap=0.0,
        mip_gap_abs=None,
        threads=int(args.threads),
        output_flag=0,
        mip_focus=1,
        heuristics=-1.0,
        presolve=2,
        cuts=2,
        symmetry=2,
        return_solution_details=True,
        fixed_binary_values=fixed_binary_values,
    )
    paper_res_ordered = solve_fixed_combo_commodity(
        instance_name=args.instance,
        scenarios=[scen],
        demand_nodes=demand_nodes,
        open_depots=open_depots,
        time_limit=float(args.time_limit),
        mip_gap=0.0,
        mip_gap_abs=None,
        threads=int(args.threads),
        output_flag=0,
        mip_focus=1,
        heuristics=-1.0,
        presolve=2,
        cuts=2,
        symmetry=2,
        return_solution_details=True,
        return_debug_variable_dump=True,
        fixed_binary_values=strict_fixed_binary_values,
    )

    relaxed_reconstructed_tt, relaxed_reconstructed_ds = _extract_reconstructed_solution(
        paper_res_relaxed.get("solution_details")
    )
    relaxed_reconstructed_arrival_obj = None
    if relaxed_reconstructed_tt or relaxed_reconstructed_ds:
        relaxed_reconstructed_arrival_obj = float(
            alg.calculate_arrival_times(relaxed_reconstructed_tt, relaxed_reconstructed_ds, scen, cfg)
        )

    ordered_reconstructed_tt, ordered_reconstructed_ds = _extract_reconstructed_solution(
        paper_res_ordered.get("solution_details")
    )
    ordered_reconstructed_arrival_obj = None
    if ordered_reconstructed_tt or ordered_reconstructed_ds:
        ordered_reconstructed_arrival_obj = float(
            alg.calculate_arrival_times(ordered_reconstructed_tt, ordered_reconstructed_ds, scen, cfg)
        )

    expected_flow = _expected_flow_from_walk(
        strict_meta=strict_meta,
        strict_fixed_binary_values=strict_fixed_binary_values,
    )
    ordered_flow_comparison = _compare_expected_vs_model_flow(
        expected=expected_flow,
        debug_dump=paper_res_ordered.get("debug_variable_dump"),
    )

    out_obj = {
        "instance": args.instance,
        "source_kind": source_kind,
        "source_file": os.path.abspath(args.table3_json or args.gurobi_json).replace("\\", "/"),
        "scenario_index": int(s),
        "open_depots_internal": [int(x) for x in open_depots],
        "explicit_solution": {
            "tt": {str(int(k)): [int(x) for x in v] for k, v in sorted(tt.items())},
            "ds": {str(int(k)): [int(x) for x in v] for k, v in sorted(ds.items())},
        },
        "arrival_sim": {
            "objective": float(arrival_obj),
        },
        "arrival_sim_on_model_normalized_walk": {
            "objective": float(strict_arrival_obj),
            "tt": {str(int(k)): [int(x) for x in v] for k, v in sorted(strict_tt.items())},
            "ds": {str(int(k)): [int(x) for x in v] for k, v in sorted(ds.items())},
        },
        "derived_from_explicit_solution": {
            **_summarize_explicit_paths(scen, demand_nodes, open_depots, tt, ds),
            **_group_fixed_assignments(fixed_binary_values, open_depots),
        },
        "strict_order_fixed_binaries_derived_from_explicit_solution": {
            **strict_meta,
            **_group_fixed_assignments(strict_fixed_binary_values, open_depots),
        },
        "paper_exact_fixed_primary_binaries_only": {
            "status": paper_res_relaxed.get("status"),
            "status_name": paper_res_relaxed.get("status_name"),
            "objective": paper_res_relaxed.get("objective"),
            "best_bound": paper_res_relaxed.get("best_bound"),
            "runtime_sec": paper_res_relaxed.get("runtime_sec"),
            "objective_components_incumbent": paper_res_relaxed.get("objective_components_incumbent"),
            "solution_details": paper_res_relaxed.get("solution_details"),
        },
        "paper_exact_fixed_primary_and_order_binaries": {
            "status": paper_res_ordered.get("status"),
            "status_name": paper_res_ordered.get("status_name"),
            "objective": paper_res_ordered.get("objective"),
            "best_bound": paper_res_ordered.get("best_bound"),
            "runtime_sec": paper_res_ordered.get("runtime_sec"),
            "objective_components_incumbent": paper_res_ordered.get("objective_components_incumbent"),
            "solution_details": paper_res_ordered.get("solution_details"),
            "debug_variable_dump": paper_res_ordered.get("debug_variable_dump"),
        },
        "solver_reconstructed_from_relaxed_fixing": {
            "tt": {str(int(k)): [int(x) for x in v] for k, v in sorted(relaxed_reconstructed_tt.items())},
            "ds": {str(int(k)): [int(x) for x in v] for k, v in sorted(relaxed_reconstructed_ds.items())},
            "arrival_sim_objective": relaxed_reconstructed_arrival_obj,
            "truck_walk_exact_match_vs_input": {
                str(int(k)): (relaxed_reconstructed_tt.get(int(k), []) == tt.get(int(k), []))
                for k in sorted(int(x) for x in open_depots)
            },
            "drone_seq_exact_match_vs_input": {
                str(int(k)): (relaxed_reconstructed_ds.get(int(k), []) == ds.get(int(k), []))
                for k in sorted(int(x) for x in open_depots)
            },
        },
        "solver_reconstructed_from_order_fixing": {
            "tt": {str(int(k)): [int(x) for x in v] for k, v in sorted(ordered_reconstructed_tt.items())},
            "ds": {str(int(k)): [int(x) for x in v] for k, v in sorted(ordered_reconstructed_ds.items())},
            "arrival_sim_objective": ordered_reconstructed_arrival_obj,
            "truck_walk_exact_match_vs_model_normalized_input": {
                str(int(k)): (ordered_reconstructed_tt.get(int(k), []) == strict_tt.get(int(k), []))
                for k in sorted(int(x) for x in open_depots)
            },
            "drone_seq_exact_match_vs_input": {
                str(int(k)): (ordered_reconstructed_ds.get(int(k), []) == ds.get(int(k), []))
                for k in sorted(int(x) for x in open_depots)
            },
        },
        "ordered_flow_accounting": {
            "expected_from_explicit_walk": expected_flow,
            "comparison_to_model_f": ordered_flow_comparison,
        },
        "gap": {
            "raw_arrival_minus_relaxed_paper": (
                None
                if paper_res_relaxed.get("objective") is None
                else float(arrival_obj) - float(paper_res_relaxed.get("objective"))
            ),
            "raw_arrival_minus_ordered_paper": (
                None
                if paper_res_ordered.get("objective") is None
                else float(arrival_obj) - float(paper_res_ordered.get("objective"))
            ),
            "normalized_arrival_minus_ordered_paper": (
                None
                if paper_res_ordered.get("objective") is None
                else float(strict_arrival_obj) - float(paper_res_ordered.get("objective"))
            ),
            "relaxed_reconstructed_arrival_minus_raw_arrival": (
                None
                if relaxed_reconstructed_arrival_obj is None
                else float(relaxed_reconstructed_arrival_obj) - float(arrival_obj)
            ),
            "ordered_reconstructed_arrival_minus_normalized_arrival": (
                None
                if ordered_reconstructed_arrival_obj is None
                else float(ordered_reconstructed_arrival_obj) - float(strict_arrival_obj)
            ),
            "ordered_reconstructed_arrival_minus_ordered_paper": (
                None
                if ordered_reconstructed_arrival_obj is None or paper_res_ordered.get("objective") is None
                else float(ordered_reconstructed_arrival_obj) - float(paper_res_ordered.get("objective"))
            ),
        },
        "notes": [
            "arrival_sim uses the explicit TTk/DSk directly on one scenario.",
            "arrival_sim_on_model_normalized_walk uses the deduplicated and depot-closed walk that is actually encoded into the fixed-order MIP.",
            "paper_exact_fixed_primary_binaries_only fixes only y/c/z and leaves order hints free.",
            "paper_exact_fixed_primary_and_order_binaries also fixes w/a/b derived from the explicit walk.",
            "If the explicit truck walk repeats directed arcs, binary y collapses multiplicity and the two evaluations are not strictly the same object.",
            "If the explicit truck walk does not end at its depot, the debug pipeline auto-closes it before building y.",
        ],
    }

    out_path = args.out.strip()
    if not out_path:
        src_tag = source_kind
        out_path = os.path.join(
            "outputs",
            f"debug_explicit_scores_{args.instance}_{src_tag}_scenario{s}.json",
        )

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, ensure_ascii=False, indent=2)

    print(f"Saved debug report: {out_path}")
    print(f"arrival_sim raw objective = {arrival_obj}")
    print(f"arrival_sim normalized-for-model objective = {strict_arrival_obj}")
    print(f"paper_exact relaxed(y/c/z) objective = {paper_res_relaxed.get('objective')}")
    print(f"paper_exact ordered(y/c/z/w/a/b) objective = {paper_res_ordered.get('objective')}")


if __name__ == "__main__":
    main()
