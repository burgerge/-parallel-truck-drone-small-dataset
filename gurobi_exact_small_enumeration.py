"""
Enumerative Gurobi solver for small stochastic truck-drone instances.

Key idea:
1) Enumerate all open-depot combinations C(|W|, p).
2) For each fixed combination, solve only the second-stage routing/scheduling MIP.
3) Record objective for each combination and pick the best one.

Model notes:
- Uses a paper-oriented commodity-flow second-stage model (Eq.4-44 style)
  with fixed SPT drone ordering.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
from collections import Counter
from itertools import combinations
from typing import Dict, List, Optional, Sequence, Set, Tuple

import gurobipy as gp
from gurobipy import GRB

import alg
from main import load_instance_data
from paper_eval_common import parse_instance_names


def _status_name(code: int) -> str:
    mapping = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.CUTOFF: "CUTOFF",
        GRB.ITERATION_LIMIT: "ITERATION_LIMIT",
        GRB.NODE_LIMIT: "NODE_LIMIT",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.SOLUTION_LIMIT: "SOLUTION_LIMIT",
        GRB.INTERRUPTED: "INTERRUPTED",
        GRB.NUMERIC: "NUMERIC",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
    }
    return mapping.get(code, f"STATUS_{code}")


def _drone_reachable(scen, depot: int, demand: int) -> bool:
    reachable = scen.drone_reachability.get(depot, [])
    if not reachable:
        return not math.isinf(scen.drone_times.get(depot, {}).get(demand, math.inf))
    return demand in set(reachable)


def _time_horizons(
    scenarios,
    demand_nodes: List[int],
    depots: List[int],
    drone_reachable: Dict[Tuple[int, int], Set[int]],
) -> Tuple[float, float, float]:
    n = max(1, len(demand_nodes))
    max_truck_arc = 1.0
    max_drone_rt = 1.0

    for s, scen in enumerate(scenarios):
        for i in [*demand_nodes, *depots]:
            for j in [*demand_nodes, *depots]:
                t = scen.truck_times.get(i, {}).get(j, math.inf)
                if not math.isinf(t):
                    max_truck_arc = max(max_truck_arc, float(t))
        for k in depots:
            for i in drone_reachable[(s, k)]:
                rt = scen.drone_times.get(k, {}).get(i, math.inf)
                if not math.isinf(rt):
                    max_drone_rt = max(max_drone_rt, float(rt))

    max_arc_time = max(max_truck_arc, max_drone_rt)
    truck_horizon = n * max_truck_arc
    drone_horizon = n * max_drone_rt
    return float(max_arc_time), float(truck_horizon), float(drone_horizon)


def _map_x_to_base(x_open: Sequence[int], depot_base_map: Dict[int, int]) -> List[int]:
    return [int(depot_base_map.get(int(k), int(k))) for k in x_open]


def _map_x_to_idx(x_open: Sequence[int], candidate_order: Sequence[int]) -> List[int]:
    pos = {int(k): i for i, k in enumerate(candidate_order)}
    return sorted(pos[int(k)] for k in x_open)


def _euler_walk_from_arcs(arcs: Sequence[Tuple[int, int]], start: int) -> Tuple[List[int], List[Tuple[int, int]]]:
    """
    Build one depot-anchored Euler walk from active directed arcs.
    Returns (walk_nodes, unused_arcs). If no active arc leaves start, walk is [start].
    """
    if not arcs:
        return [int(start)], []

    # deterministic traversal order for reproducible dumps
    out_map: Dict[int, List[int]] = {}
    for i, j in sorted((int(i), int(j)) for (i, j) in arcs):
        out_map.setdefault(i, []).append(j)
    for i in out_map:
        out_map[i].sort(reverse=True)

    stack: List[int] = [int(start)]
    circuit: List[int] = []
    while stack:
        v = stack[-1]
        out_list = out_map.get(v, [])
        if out_list:
            nxt = out_list.pop()
            stack.append(int(nxt))
        else:
            circuit.append(stack.pop())

    walk = list(reversed(circuit))
    if not walk:
        walk = [int(start)]

    used_edges = [(walk[idx], walk[idx + 1]) for idx in range(len(walk) - 1)]
    arc_counter = Counter((int(i), int(j)) for (i, j) in arcs)
    used_counter = Counter(used_edges)
    unused: List[Tuple[int, int]] = []
    for edge, cnt in arc_counter.items():
        rem = int(cnt - used_counter.get(edge, 0))
        if rem > 0:
            unused.extend([edge] * rem)
    return walk, unused


def _walk_from_model_hints(
    arcs: Sequence[Tuple[int, int]],
    start: int,
    first_departure_arcs: Sequence[Tuple[int, int]],
    successor_links: Sequence[Tuple[int, int, int]],
) -> Tuple[List[int], List[Tuple[int, int]]]:
    """
    Reconstruct a truck walk from model order hints:
    - w-arcs (first departure from demand node)
    - b-links (immediate successor arc relation)

    This is intended for reproducible post-hoc evaluation/inspection and keeps
    deterministic fallback behaviour if hints are incomplete.
    """
    if not arcs:
        return [int(start)], []

    out_map: Dict[int, List[int]] = {}
    for i, j in sorted((int(i), int(j)) for (i, j) in arcs):
        out_map.setdefault(i, []).append(j)
    for i in out_map:
        out_map[i] = sorted(out_map[i])

    w_map: Dict[int, int] = {}
    for i, j in sorted((int(i), int(j)) for (i, j) in first_departure_arcs):
        w_map[int(i)] = int(j)

    b_map: Dict[Tuple[int, int], int] = {}
    for i, j, l in sorted((int(i), int(j), int(l)) for (i, j, l) in successor_links):
        b_map[(int(i), int(j))] = int(l)

    total = len(arcs)
    used = Counter()  # edge -> count
    leave_count = Counter()  # node -> number of departures in reconstructed walk

    walk: List[int] = [int(start)]
    prev: Optional[int] = None
    cur = int(start)
    max_steps = max(10, 3 * total + 10)

    for _ in range(max_steps):
        if sum(used.values()) >= total:
            break
        cand_all = out_map.get(int(cur), [])
        if not cand_all:
            break
        cand_unused = [int(v) for v in cand_all if used[(int(cur), int(v))] <= 0]
        if not cand_unused:
            break

        nxt: Optional[int] = None
        # Prefer model hint: first departure arc on first leave from a demand node.
        hint_w = w_map.get(int(cur))
        if leave_count[int(cur)] == 0 and hint_w is not None and hint_w in cand_unused:
            nxt = int(hint_w)

        # Then prefer immediate successor hint from b-links.
        if nxt is None and prev is not None:
            hint_b = b_map.get((int(prev), int(cur)))
            if hint_b is not None and hint_b in cand_unused:
                nxt = int(hint_b)

        # Deterministic fallback.
        if nxt is None:
            nxt = int(sorted(cand_unused)[0])

        used[(int(cur), int(nxt))] += 1
        leave_count[int(cur)] += 1
        walk.append(int(nxt))
        prev, cur = int(cur), int(nxt)

    arc_counter = Counter((int(i), int(j)) for (i, j) in arcs)
    unused: List[Tuple[int, int]] = []
    for edge, cnt in arc_counter.items():
        rem = int(cnt - used.get(edge, 0))
        if rem > 0:
            unused.extend([edge] * rem)
    return walk, unused


def _arrival_sim_from_solution_routes(
    scenarios: Sequence,
    scenario_routes: Sequence[Dict],
    num_open_depots: int,
) -> Dict[str, object]:
    """
    Evaluate explicit TT/DS routes by the heuristic arrival simulation metric.
    """
    cfg = alg.HeuristicConfig(
        num_depots_to_open=max(1, int(num_open_depots)),
        num_scenarios=max(1, len(scenarios)),
        seed=123,
        k_max=7,
        l_max=6,
        i_max=5,
        drone_time_is_roundtrip=True,
        normalize_by_num_demands=True,
        strict_feasibility=True,
    )

    vals: List[float] = []
    for rec in scenario_routes:
        s = int(rec.get("scenario_index", 0))
        if s < 0 or s >= len(scenarios):
            continue
        tt: Dict[int, List[int]] = {}
        ds: Dict[int, List[int]] = {}
        for dep in rec.get("depots", []):
            k = int(dep.get("depot_internal"))
            walk = dep.get("truck_walk_eval_preferred", []) or dep.get("truck_walk_model_order", []) or dep.get("truck_walk", [])
            tt[k] = [int(x) for x in (walk or [k])]
            ds[k] = [int(x) for x in dep.get("drone_sequence_spt", [])]
        vals.append(float(alg.calculate_arrival_times(tt, ds, scenarios[s], cfg)))

    if not vals:
        return {"weighted_avg": None, "scenario_values": []}
    return {
        "weighted_avg": float(sum(vals) / len(vals)),
        "scenario_values": vals,
    }


def _prepare_data(
    instance_name: str,
    instances_root: str,
    max_scenarios: int,
    max_demands: int,
):
    folder = os.path.join(instances_root, instance_name)
    if not os.path.exists(folder):
        raise FileNotFoundError(f"Instance folder not found: {folder}")

    base_instance, scenarios = load_instance_data(folder)
    if not scenarios:
        raise ValueError("No scenarios loaded from t.txt.")

    demand_nodes = list(base_instance.demand_nodes)
    candidate_depots = list(base_instance.candidate_depots)
    if max_demands > 0:
        demand_nodes = demand_nodes[: max_demands]
    if max_scenarios > 0:
        scenarios = scenarios[: max_scenarios]

    demand_set = set(demand_nodes)
    base_instance.demand_nodes = list(demand_nodes)
    base_instance.drone_reachability = {
        k: [i for i in v if i in demand_set]
        for k, v in getattr(base_instance, "drone_reachability", {}).items()
    }
    for scen in scenarios:
        scen.demand_nodes = list(demand_nodes)
        scen.drone_reachability = {
            k: [i for i in v if i in demand_set]
            for k, v in scen.drone_reachability.items()
        }

    return base_instance, scenarios, demand_nodes, candidate_depots


def solve_fixed_combo_commodity(
    instance_name: str,
    scenarios: List,
    demand_nodes: List[int],
    open_depots: Sequence[int],
    time_limit: float,
    mip_gap: float,
    mip_gap_abs: Optional[float],
    threads: int,
    output_flag: int,
    mip_focus: int,
    heuristics: float,
    presolve: int,
    cuts: int,
    symmetry: int,
    return_solution_details: bool = False,
    return_debug_variable_dump: bool = False,
    fixed_binary_values: Optional[Dict[str, Set[Tuple[int, ...]]]] = None,
    arrival_rescore: bool = False,
    arrival_pool_solutions: int = 16,
    arrival_pool_search_mode: int = 2,
    arrival_pool_gap: Optional[float] = None,
) -> Dict:
    """
    Paper-oriented commodity-flow second-stage model for fixed open depots.

    This implementation aligns with the paper's Eq. (4)-(44) structure:
    - truck assignment c, drone assignment z
    - depot-indexed truck arc variables y
    - first/last departure binaries w/a
    - immediate-successor binaries b
    - commodity flow f on expanded network E' with sink node
    - auxiliary flow variables phi/psi/sigma and visit counters u/h
    - fixed SPT drone-order arrival constraints (Eq. 5 logic)
    """
    depots = sorted(int(k) for k in open_depots)
    num_s = len(scenarios)
    if num_s <= 0:
        raise ValueError("No scenarios to optimize.")

    drone_reachable = {
        (s, k): {i for i in demand_nodes if _drone_reachable(scen, k, i)}
        for s, scen in enumerate(scenarios)
        for k in depots
    }

    max_arc_time, truck_horizon, drone_horizon = _time_horizons(
        scenarios, demand_nodes, depots, drone_reachable
    )
    m_time = max(truck_horizon, drone_horizon) + max_arc_time
    n_demands = len(demand_nodes)
    if n_demands <= 0:
        raise ValueError("No demand nodes to optimize.")
    m_flow = float(n_demands + 1)

    model = gp.Model(f"enum_paper_{instance_name}_{'_'.join(map(str, depots))}")
    model.Params.OutputFlag = int(output_flag)
    model.Params.TimeLimit = float(time_limit)
    model.Params.MIPGap = float(mip_gap)
    if mip_gap_abs is not None and mip_gap_abs >= 0.0:
        model.Params.MIPGapAbs = float(mip_gap_abs)
    if threads > 0:
        model.Params.Threads = int(threads)
    if mip_focus in (0, 1, 2, 3):
        model.Params.MIPFocus = int(mip_focus)
    if heuristics >= 0.0:
        model.Params.Heuristics = float(heuristics)
    if presolve in (-1, 0, 1, 2):
        model.Params.Presolve = int(presolve)
    if cuts in (-1, 0, 1, 2, 3):
        model.Params.Cuts = int(cuts)
    if symmetry in (-1, 0, 1, 2):
        model.Params.Symmetry = int(symmetry)
    if bool(arrival_rescore):
        if int(arrival_pool_search_mode) in (0, 1, 2):
            model.Params.PoolSearchMode = int(arrival_pool_search_mode)
        if int(arrival_pool_solutions) > 0:
            model.Params.PoolSolutions = int(arrival_pool_solutions)
        if arrival_pool_gap is not None and float(arrival_pool_gap) >= 0.0:
            model.Params.PoolGap = float(arrival_pool_gap)

    sink = -1
    dset = set(demand_nodes)
    fixed_y_on: Optional[Set[Tuple[int, int, int, int]]] = None
    fixed_z_on: Optional[Set[Tuple[int, int, int]]] = None
    fixed_c_on: Optional[Set[Tuple[int, int, int]]] = None
    fixed_w_on: Optional[Set[Tuple[int, int, int, int]]] = None
    fixed_a_on: Optional[Set[Tuple[int, int, int, int]]] = None
    fixed_b_on: Optional[Set[Tuple[int, int, int, int]]] = None
    if fixed_binary_values:
        y_on_raw = fixed_binary_values.get("y_on")
        z_on_raw = fixed_binary_values.get("z_on")
        c_on_raw = fixed_binary_values.get("c_on")
        w_on_raw = fixed_binary_values.get("w_on")
        a_on_raw = fixed_binary_values.get("a_on")
        b_on_raw = fixed_binary_values.get("b_on")
        if y_on_raw is not None:
            fixed_y_on = set((int(t[0]), int(t[1]), int(t[2]), int(t[3])) for t in y_on_raw)
        if z_on_raw is not None:
            fixed_z_on = set((int(t[0]), int(t[1]), int(t[2])) for t in z_on_raw)
        if c_on_raw is not None:
            fixed_c_on = set((int(t[0]), int(t[1]), int(t[2])) for t in c_on_raw)
        if w_on_raw is not None:
            fixed_w_on = set((int(t[0]), int(t[1]), int(t[2]), int(t[3])) for t in w_on_raw)
        if a_on_raw is not None:
            fixed_a_on = set((int(t[0]), int(t[1]), int(t[2]), int(t[3])) for t in a_on_raw)
        if b_on_raw is not None:
            fixed_b_on = set((int(t[0]), int(t[1]), int(t[2]), int(t[3])) for t in b_on_raw)

    # Per-scenario arc bookkeeping.
    arcs_E: Dict[int, List[Tuple[int, int]]] = {}
    arcs_Ep: Dict[int, List[Tuple[int, int]]] = {}
    out_E: Dict[int, Dict[int, List[int]]] = {}
    in_E: Dict[int, Dict[int, List[int]]] = {}
    out_Ep: Dict[int, Dict[int, List[int]]] = {}
    in_Ep: Dict[int, Dict[int, List[int]]] = {}
    y_arcs_sk: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
    y_in_sk: Dict[Tuple[int, int, int], List[int]] = {}
    y_out_sk: Dict[Tuple[int, int, int], List[int]] = {}
    arc_ks: Dict[Tuple[int, int, int], List[int]] = {}
    preds_j: Dict[Tuple[int, int], List[int]] = {}
    succs_j: Dict[Tuple[int, int], List[int]] = {}
    b_out_pairs: Dict[Tuple[int, int, int], List[int]] = {}
    drone_orders: Dict[Tuple[int, int], List[int]] = {}

    all_nodes = list(dict.fromkeys(demand_nodes + depots))
    all_node_set = set(all_nodes)
    for s, scen in enumerate(scenarios):
        e_list: List[Tuple[int, int]] = []
        out_e = {i: [] for i in all_nodes}
        in_e = {i: [] for i in all_nodes}
        for i in all_nodes:
            row = scen.truck_times.get(i, {})
            for j in all_nodes:
                if i == j:
                    continue
                t = row.get(j, math.inf)
                if math.isinf(t):
                    continue
                e_list.append((i, j))
                out_e[i].append(j)
                in_e[j].append(i)
        arcs_E[s] = e_list
        out_E[s] = out_e
        in_E[s] = in_e

        # Build route arcs for each open depot route-k (only depot k and demand nodes).
        e_union: Set[Tuple[int, int]] = set()
        for k in depots:
            nodes_k = set(demand_nodes + [k])
            a_k: List[Tuple[int, int]] = []
            for (i, j) in e_list:
                if i in nodes_k and j in nodes_k:
                    a_k.append((i, j))
                    e_union.add((i, j))
            y_arcs_sk[(s, k)] = a_k

            for i in nodes_k:
                y_in_sk[(s, k, i)] = []
                y_out_sk[(s, k, i)] = []
            for (i, j) in a_k:
                y_out_sk[(s, k, i)].append(j)
                y_in_sk[(s, k, j)].append(i)
                arc_ks.setdefault((s, i, j), []).append(k)

            drone_orders[(s, k)] = sorted(
                drone_reachable[(s, k)],
                key=lambda i: float(scen.drone_times.get(k, {}).get(i, math.inf)),
            )

        # Expanded edge set E' (truck arcs in any route + sink arcs from all N nodes).
        ep = sorted(e_union)
        for i in all_nodes:
            ep.append((i, sink))
        arcs_Ep[s] = ep
        out_ep = {i: [] for i in all_nodes + [sink]}
        in_ep = {i: [] for i in all_nodes + [sink]}
        for (i, j) in ep:
            out_ep.setdefault(i, []).append(j)
            in_ep.setdefault(j, []).append(i)
        out_Ep[s] = out_ep
        in_Ep[s] = in_ep

        for j in demand_nodes:
            preds = sorted([i for i in in_e.get(j, []) if i in all_node_set and i != sink])
            succs = sorted([l for l in out_e.get(j, []) if l in all_node_set and l != j])
            preds_j[(s, j)] = preds
            succs_j[(s, j)] = succs
            for l in succs:
                b_out_pairs[(s, j, l)] = preds.copy()

    # Decision variables
    c = model.addVars(range(num_s), depots, demand_nodes, vtype=GRB.BINARY, name="c")
    z = model.addVars(range(num_s), depots, demand_nodes, vtype=GRB.BINARY, name="z")

    y_full_keys: List[Tuple[int, int, int, int]] = []
    for s in range(num_s):
        for k in depots:
            for (i, j) in y_arcs_sk[(s, k)]:
                y_full_keys.append((int(s), int(k), int(i), int(j)))
    y_full_key_set = set(y_full_keys)

    if fixed_y_on is not None:
        bad = [t for t in fixed_y_on if t not in y_full_key_set]
        if bad:
            raise ValueError(f"fixed y_on contains invalid keys, sample={bad[:5]}")
        y_keys = [t for t in y_full_keys if t in fixed_y_on]
    else:
        y_keys = list(y_full_keys)
    y_key_set = set(y_keys)

    y_var_arcs_sk: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
    for s in range(num_s):
        for k in depots:
            if fixed_y_on is None:
                y_var_arcs_sk[(s, k)] = list(y_arcs_sk[(s, k)])
            else:
                y_var_arcs_sk[(s, k)] = [
                    (int(i), int(j))
                    for (i, j) in y_arcs_sk[(s, k)]
                    if (int(s), int(k), int(i), int(j)) in y_key_set
                ]

    w_keys: List[Tuple[int, int, int, int]] = []
    a_keys: List[Tuple[int, int, int, int]] = []
    for (s, k, i, j) in y_keys:
        if int(i) in dset:
            w_keys.append((int(s), int(k), int(i), int(j)))
            a_keys.append((int(s), int(k), int(i), int(j)))

    if fixed_y_on is None:
        y = model.addVars(y_keys, vtype=GRB.BINARY, name="y")
    else:
        # Fixed-route mode: only y=1 arcs are instantiated, with fixed bounds.
        y = model.addVars(y_keys, lb=1.0, ub=1.0, vtype=GRB.BINARY, name="y")
    w = model.addVars(w_keys, vtype=GRB.BINARY, name="w")
    a_last = model.addVars(a_keys, vtype=GRB.BINARY, name="a_last")

    # Speed-up in fixed-route mode: prune b-index space to active y arcs only.
    if fixed_y_on is not None:
        active_arc_any = {(int(s), int(i), int(j)) for (s, _k, i, j) in y_keys}
        b_out_pairs = {}
        for s in range(num_s):
            for j in demand_nodes:
                preds = [int(i) for i in preds_j[(s, j)] if (int(s), int(i), int(j)) in active_arc_any]
                succs = [int(l) for l in succs_j[(s, j)] if (int(s), int(j), int(l)) in active_arc_any]
                preds_j[(s, j)] = preds
                succs_j[(s, j)] = succs
                for l in succs:
                    b_out_pairs[(int(s), int(j), int(l))] = preds.copy()

    b_keys: List[Tuple[int, int, int, int]] = []
    b_keys_by_s: Dict[int, List[Tuple[int, int, int]]] = {int(s): [] for s in range(num_s)}
    for s in range(num_s):
        for j in demand_nodes:
            for i in preds_j[(s, j)]:
                for l in succs_j[(s, j)]:
                    b_keys.append((int(s), int(i), int(j), int(l)))
                    b_keys_by_s[int(s)].append((int(i), int(j), int(l)))
    b = model.addVars(b_keys, vtype=GRB.BINARY, name="b")

    f_keys = []
    for s in range(num_s):
        f_keys.extend((s, i, j) for (i, j) in arcs_Ep[s])
    f = model.addVars(f_keys, lb=0.0, vtype=GRB.CONTINUOUS, name="f")

    p = model.addVars(range(num_s), demand_nodes, lb=0.0, vtype=GRB.CONTINUOUS, name="p")
    phi = model.addVars(range(num_s), demand_nodes, lb=0.0, vtype=GRB.CONTINUOUS, name="phi")
    psi = model.addVars(range(num_s), demand_nodes, lb=0.0, vtype=GRB.CONTINUOUS, name="psi")
    sigma = model.addVars(range(num_s), demand_nodes, lb=0.0, vtype=GRB.CONTINUOUS, name="sigma")
    u_vis = model.addVars(range(num_s), depots, demand_nodes, lb=0.0, vtype=GRB.CONTINUOUS, name="u_vis")
    h = model.addVars(range(num_s), depots, demand_nodes, lb=0.0, vtype=GRB.CONTINUOUS, name="h")

    # Optional: evaluate a fixed route/assignment plan under the paper-exact model
    # by fixing primary binary decisions y/z/c.
    if fixed_binary_values:
        if fixed_z_on is not None:
            z_key_set = set((int(s), int(k), int(i)) for s in range(num_s) for k in depots for i in demand_nodes)
            bad = [t for t in fixed_z_on if t not in z_key_set]
            if bad:
                raise ValueError(f"fixed z_on contains invalid keys, sample={bad[:5]}")
            for s in range(num_s):
                for k in depots:
                    for i in demand_nodes:
                        val = 1.0 if (int(s), int(k), int(i)) in fixed_z_on else 0.0
                        z[s, k, i].LB = val
                        z[s, k, i].UB = val

        if fixed_c_on is not None:
            c_key_set = set((int(s), int(k), int(i)) for s in range(num_s) for k in depots for i in demand_nodes)
            bad = [t for t in fixed_c_on if t not in c_key_set]
            if bad:
                raise ValueError(f"fixed c_on contains invalid keys, sample={bad[:5]}")
            for s in range(num_s):
                for k in depots:
                    for i in demand_nodes:
                        val = 1.0 if (int(s), int(k), int(i)) in fixed_c_on else 0.0
                        c[s, k, i].LB = val
                        c[s, k, i].UB = val

        if fixed_w_on is not None:
            w_key_set = set((int(t[0]), int(t[1]), int(t[2]), int(t[3])) for t in w_keys)
            bad = [t for t in fixed_w_on if t not in w_key_set]
            if bad:
                raise ValueError(f"fixed w_on contains invalid keys, sample={bad[:5]}")
            for (s, k, i, j) in w_keys:
                val = 1.0 if (int(s), int(k), int(i), int(j)) in fixed_w_on else 0.0
                w[s, k, i, j].LB = val
                w[s, k, i, j].UB = val

        if fixed_a_on is not None:
            a_key_set = set((int(t[0]), int(t[1]), int(t[2]), int(t[3])) for t in a_keys)
            bad = [t for t in fixed_a_on if t not in a_key_set]
            if bad:
                raise ValueError(f"fixed a_on contains invalid keys, sample={bad[:5]}")
            for (s, k, i, j) in a_keys:
                val = 1.0 if (int(s), int(k), int(i), int(j)) in fixed_a_on else 0.0
                a_last[s, k, i, j].LB = val
                a_last[s, k, i, j].UB = val

        if fixed_b_on is not None:
            b_key_set = set((int(t[0]), int(t[1]), int(t[2]), int(t[3])) for t in b_keys)
            bad = [t for t in fixed_b_on if t not in b_key_set]
            if bad:
                raise ValueError(f"fixed b_on contains invalid keys, sample={bad[:5]}")
            for (s, i, j, l) in b_keys:
                val = 1.0 if (int(s), int(i), int(j), int(l)) in fixed_b_on else 0.0
                b[s, i, j, l].LB = val
                b[s, i, j, l].UB = val

    for s, scen in enumerate(scenarios):
        # Eq. (5): Drone arrival time with SPT order per depot.
        for k in depots:
            order = drone_orders[(s, k)]
            for pos, i in enumerate(order):
                prior = order[:pos]
                prior_rt_sum = gp.quicksum(float(scen.drone_times[k][j]) * z[s, k, j] for j in prior)
                rt_i = float(scen.drone_times[k][i])
                ow_i = rt_i / 2.0
                model.addConstr(
                    p[s, i] >= ow_i + prior_rt_sum - m_time * (1 - z[s, k, i]),
                    name=f"eq5_p_s{s}_k{k}_i{i}",
                )

        # Eq. (6): each demand assigned exactly once (truck or drone).
        for i in demand_nodes:
            model.addConstr(
                gp.quicksum(c[s, k, i] + z[s, k, i] for k in depots) == 1,
                name=f"eq6_cover_s{s}_i{i}",
            )

        # Eq. (8): drone assignment only if endurance-feasible.
        for k in depots:
            for i in demand_nodes:
                if i not in drone_reachable[(s, k)]:
                    model.addConstr(z[s, k, i] == 0, name=f"eq8_z_reach_s{s}_k{k}_i{i}")

        # Eq. (7), (9), (10), (12), (13), (14)
        for k in depots:
            # Eq. (10): at most one departure from open depot k.
            model.addConstr(
                gp.quicksum(
                    y[s, k, k, j]
                    for j in y_out_sk.get((s, k, k), [])
                    if (s, k, k, j) in y
                )
                <= 1,
                name=f"eq10_depot_depart_s{s}_k{k}",
            )

            # Eq. (12): flow conservation on N for route-k
            nodes_k = demand_nodes + [k]
            for i in nodes_k:
                out_expr = gp.quicksum(
                    y[s, k, i, j]
                    for j in y_out_sk.get((s, k, i), [])
                    if (s, k, i, j) in y
                )
                in_expr = gp.quicksum(
                    y[s, k, j, i]
                    for j in y_in_sk.get((s, k, i), [])
                    if (s, k, j, i) in y
                )
                model.addConstr(out_expr == in_expr, name=f"eq12_flow_s{s}_k{k}_i{i}")

            for i in demand_nodes:
                # Eq. (7): demand appears on route-k only if truck-assigned to k.
                in_expr = gp.quicksum(
                    y[s, k, j, i]
                    for j in y_in_sk.get((s, k, i), [])
                    if (s, k, j, i) in y
                )
                out_expr = gp.quicksum(
                    y[s, k, i, j]
                    for j in y_out_sk.get((s, k, i), [])
                    if (s, k, i, j) in y
                )
                model.addConstr(
                    in_expr + out_expr <= 2.0 * n_demands * c[s, k, i],
                    name=f"eq7_route_consistency_s{s}_k{k}_i{i}",
                )
                # Eq. (13): visit counter u
                model.addConstr(
                    in_expr == u_vis[s, k, i],
                    name=f"eq13_u_s{s}_k{k}_i{i}",
                )
                # Eq. (14): additional visits lower bound.
                model.addConstr(
                    h[s, k, i] >= u_vis[s, k, i] - 1.0,
                    name=f"eq14_h_s{s}_k{k}_i{i}",
                )

            # Eq. (9): y only if depot k is open (fixed open in enumeration => always 1).
            for (i, j) in y_var_arcs_sk[(s, k)]:
                model.addConstr(y[s, k, i, j] <= 1, name=f"eq9_open_s{s}_k{k}_i{i}_j{j}")

        # Eq. (15): depot commodity source.
        for k in depots:
            out_f_k = gp.quicksum(
                f[s, k, j]
                for j in out_Ep[s].get(k, [])
                if (s, k, j) in f
            )
            y_count = gp.quicksum(
                y[s, k, i, j]
                for (i, j) in y_arcs_sk[(s, k)]
                if (s, k, i, j) in y
            )
            model.addConstr(
                out_f_k == y_count - gp.quicksum(h[s, k, i] for i in demand_nodes),
                name=f"eq15_source_s{s}_k{k}",
            )

        # Eq. (16): demand-node flow balance.
        for i in demand_nodes:
            in_f_i = gp.quicksum(
                f[s, j, i]
                for j in in_Ep[s].get(i, [])
                if (s, j, i) in f
            )
            out_f_i = gp.quicksum(
                f[s, i, j]
                for j in out_Ep[s].get(i, [])
                if (s, i, j) in f
            )
            model.addConstr(
                in_f_i - out_f_i == 1.0 - gp.quicksum(z[s, k, i] for k in depots),
                name=f"eq16_bal_s{s}_i{i}",
            )

        # Eq. (17): sink inflow equals number of open depots.
        model.addConstr(
            gp.quicksum(
                f[s, i, sink]
                for i in all_nodes
                if (s, i, sink) in f
            )
            == float(len(depots)),
            name=f"eq17_sink_s{s}",
        )

        # Eq. (18): if no truck on arc, no commodity flow.
        for (i, j) in arcs_E[s]:
            y_on_arc = gp.quicksum(
                y[s, k, i, j] for k in arc_ks.get((s, i, j), []) if (s, k, i, j) in y
            )
            if (s, i, j) in f:
                model.addConstr(
                    f[s, i, j] <= m_flow * y_on_arc,
                    name=f"eq18_ub_s{s}_i{i}_j{j}",
                )

        # Eq. (19): if a truck traverses an arc into a demand node, flow must be positive.
        for k in depots:
            for (i, j) in y_arcs_sk[(s, k)]:
                if j not in dset:
                    continue
                if (s, i, j) in f and (s, k, i, j) in y:
                    model.addConstr(
                        f[s, i, j] >= y[s, k, i, j],
                        name=f"eq19_lb_s{s}_k{k}_i{i}_j{j}",
                    )

        # Eq. (20): sink flow from node i only if truck returns from i to some depot.
        for i in all_nodes:
            if (s, i, sink) not in f:
                continue
            back_to_depot = gp.quicksum(
                y[s, k, i, k]
                for k in depots
                if (s, k, i, k) in y
            )
            model.addConstr(
                f[s, i, sink] <= back_to_depot,
                name=f"eq20_sink_link_s{s}_i{i}",
            )

        # Eq. (21)-(22): first/last departure arc can be active only if arc used.
        for (s2, k, i, j) in w_keys:
            if s2 != s:
                continue
            model.addConstr(w[s, k, i, j] <= y[s, k, i, j], name=f"eq21_w_s{s}_k{k}_i{i}_j{j}")
        for (s2, k, i, j) in a_keys:
            if s2 != s:
                continue
            model.addConstr(
                a_last[s, k, i, j] <= y[s, k, i, j],
                name=f"eq22_a_s{s}_k{k}_i{i}_j{j}",
            )

        # Eq. (23)-(24): if node i is truck-served, exactly one first and one last departure.
        for i in demand_nodes:
            w_sum = gp.quicksum(
                w[s, k, i, j]
                for k in depots
                for j in y_out_sk.get((s, k, i), [])
                if (s, k, i, j) in w
            )
            a_sum = gp.quicksum(
                a_last[s, k, i, j]
                for k in depots
                for j in y_out_sk.get((s, k, i), [])
                if (s, k, i, j) in a_last
            )
            truck_indicator = 1.0 - gp.quicksum(z[s, k, i] for k in depots)
            model.addConstr(w_sum == truck_indicator, name=f"eq23_w_count_s{s}_i{i}")
            model.addConstr(a_sum == truck_indicator, name=f"eq24_a_count_s{s}_i{i}")

        # Eq. (25)-(31): flow-envelope constraints.
        for i in demand_nodes:
            # Eq. (25): on active last-departure arc, sigma is lower-bounded by that outgoing flow.
            for j in [jj for jj in out_E[s].get(i, []) if jj in dset]:
                a_ij_sum = gp.quicksum(
                    a_last[s, k, i, j] for k in arc_ks.get((s, i, j), []) if (s, k, i, j) in a_last
                )
                if (s, i, j) in f:
                    model.addConstr(
                        sigma[s, i] >= f[s, i, j] - m_flow * (1 - a_ij_sum),
                        name=f"eq25_sigma_s{s}_i{i}_j{j}",
                    )

            # Eq. (26): if last departure is to a depot, sigma is lower-bounded by sink flow.
            a_ikk_sum = gp.quicksum(
                a_last[s, k, i, k] for k in depots if (s, k, i, k) in a_last
            )
            if (s, i, sink) in f:
                model.addConstr(
                    sigma[s, i] >= f[s, i, sink] - m_flow * (1 - a_ikk_sum),
                    name=f"eq26_sigma_sink_s{s}_i{i}",
                )

            # Eq. (27): phi is max incoming flow.
            for j in in_E[s].get(i, []):
                if (s, j, i) in f:
                    model.addConstr(
                        phi[s, i] >= f[s, j, i],
                        name=f"eq27_phi_s{s}_i{i}_j{j}",
                    )

            # Eq. (28): first departure on demand-demand arc leaves phi-1 flow.
            for j in [jj for jj in out_E[s].get(i, []) if jj in dset]:
                w_ij_sum = gp.quicksum(
                    w[s, k, i, j] for k in arc_ks.get((s, i, j), []) if (s, k, i, j) in w
                )
                if (s, i, j) in f:
                    model.addConstr(
                        f[s, i, j] >= phi[s, i] - 1.0 - m_flow * (1 - w_ij_sum),
                        name=f"eq28_flow_first_s{s}_i{i}_j{j}",
                    )

            # Eq. (29): first departure to depot also leaves phi-1 flow via sink.
            w_ikk_sum = gp.quicksum(
                w[s, k, i, k] for k in depots if (s, k, i, k) in w
            )
            if (s, i, sink) in f:
                model.addConstr(
                    f[s, i, sink] >= phi[s, i] - 1.0 - m_flow * (1 - w_ikk_sum),
                    name=f"eq29_flow_sink_first_s{s}_i{i}",
                )

            # Eq. (30): psi upper-bounds minimum incoming flow.
            for j in in_E[s].get(i, []):
                y_ji_sum = gp.quicksum(
                    y[s, k, j, i] for k in arc_ks.get((s, j, i), []) if (s, k, j, i) in y
                )
                if (s, j, i) in f:
                    model.addConstr(
                        psi[s, i] <= f[s, j, i] + m_flow * (1 - y_ji_sum),
                        name=f"eq30_psi_s{s}_i{i}_j{j}",
                    )

            # Eq. (31): minimum outgoing <= minimum incoming.
            model.addConstr(sigma[s, i] <= psi[s, i], name=f"eq31_sigma_le_psi_s{s}_i{i}")

        # Eq. (32)-(34): b variable activation.
        for j in demand_nodes:
            for l in succs_j[(s, j)]:
                rhs_y_jl = gp.quicksum(
                    y[s, k, j, l] for k in arc_ks.get((s, j, l), []) if (s, k, j, l) in y
                )
                rhs_not_first = rhs_y_jl - gp.quicksum(
                    w[s, k, j, l] for k in arc_ks.get((s, j, l), []) if (s, k, j, l) in w
                )
                preds = b_out_pairs.get((s, j, l), [])
                if preds:
                    model.addConstr(
                        gp.quicksum(b[s, i, j, l] for i in preds if (s, i, j, l) in b) >= rhs_not_first,
                        name=f"eq34_b_activate_s{s}_j{j}_l{l}",
                    )
                for i in preds:
                    if (s, i, j, l) not in b:
                        continue
                    # Eq. (32)
                    model.addConstr(
                        b[s, i, j, l] <= rhs_y_jl,
                        name=f"eq32_b_s{s}_i{i}_j{j}_l{l}",
                    )
                    # Eq. (33)
                    rhs_y_ij = gp.quicksum(
                        y[s, k, i, j]
                        for k in arc_ks.get((s, i, j), [])
                        if (s, k, i, j) in y
                    )
                    model.addConstr(
                        b[s, i, j, l] <= rhs_y_ij,
                        name=f"eq33_b_s{s}_i{i}_j{j}_l{l}",
                    )

                    # Eq. (35)-(38): flow continuity along consecutive arcs.
                    if l in dset and (s, j, l) in f and (s, i, j) in f:
                        model.addConstr(
                            f[s, j, l] >= f[s, i, j] - m_flow * (1 - b[s, i, j, l]),
                            name=f"eq35_flow_s{s}_i{i}_j{j}_l{l}",
                        )
                        model.addConstr(
                            f[s, j, l] <= f[s, i, j] + m_flow * (1 - b[s, i, j, l]),
                            name=f"eq36_flow_s{s}_i{i}_j{j}_l{l}",
                        )
                    elif l in depots and (s, j, sink) in f and (s, i, j) in f:
                        model.addConstr(
                            f[s, j, sink] >= f[s, i, j] - m_flow * (1 - b[s, i, j, l]),
                            name=f"eq37_flow_sink_s{s}_i{i}_j{j}_l{l}",
                        )
                        model.addConstr(
                            f[s, j, sink] <= f[s, i, j] + m_flow * (1 - b[s, i, j, l]),
                            name=f"eq38_flow_sink_s{s}_i{i}_j{j}_l{l}",
                        )

    # Objective Eq. (4): average arrival time.
    truck_flow_component = gp.quicksum(
        float(scenarios[s].truck_times[i][j]) * f[s, i, j]
        for s in range(num_s)
        for (i, j) in arcs_E[s]
        if (s, i, j) in f
    )
    # Exclude the final return-to-depot arc from the subtraction term.
    # The commodity-flow part already captures cumulative arrival times to
    # truck-served demand nodes; subtracting depot-return arcs would
    # incorrectly reduce the objective by travel time that occurs after the
    # last beneficiary is first reached.
    truck_arc_component = gp.quicksum(
        float(scenarios[s].truck_times[i][j]) * y[s, k, i, j]
        for (s, k, i, j) in y_keys
        if int(j) not in depots
    )
    drone_component = gp.quicksum(p[s, i] for s in range(num_s) for i in demand_nodes)
    coeff = 1.0 / (num_s * n_demands)
    model.setObjective(coeff * (truck_flow_component - truck_arc_component + drone_component), GRB.MINIMIZE)
    model.optimize()

    def _var_val(var, use_pool: bool) -> float:
        if use_pool:
            try:
                return float(var.Xn)
            except Exception:
                return float(var.X)
        return float(var.X)

    def _extract_solution_details(solution_number: int = 0, use_pool: bool = False) -> Dict:
        if use_pool:
            try:
                model.Params.SolutionNumber = int(solution_number)
            except Exception:
                pass

        scenario_routes: List[Dict] = []
        for s in range(num_s):
            scen_rec: Dict = {"scenario_index": int(s), "depots": []}
            for k in depots:
                active_arcs = [
                    (int(i), int(j))
                    for (i, j) in y_var_arcs_sk[(s, k)]
                    if (s, k, i, j) in y and _var_val(y[s, k, i, j], use_pool) > 0.5
                ]
                walk_euler, unused_euler = _euler_walk_from_arcs(active_arcs, start=k)
                active_set = set((int(i), int(j)) for (i, j) in active_arcs)
                w_arcs = [
                    (int(i), int(j))
                    for (i, j) in y_var_arcs_sk[(s, k)]
                    if (s, k, i, j) in w and _var_val(w[s, k, i, j], use_pool) > 0.5
                ]
                a_arcs = [
                    (int(i), int(j))
                    for (i, j) in y_var_arcs_sk[(s, k)]
                    if (s, k, i, j) in a_last and _var_val(a_last[s, k, i, j], use_pool) > 0.5
                ]
                b_links = [
                    (int(i), int(j), int(l))
                    for (i, j, l) in b_keys_by_s.get(int(s), [])
                    if (s, i, j, l) in b
                    and _var_val(b[s, i, j, l], use_pool) > 0.5
                    and (int(i), int(j)) in active_set
                    and (int(j), int(l)) in active_set
                ]
                walk_model, unused_model = _walk_from_model_hints(
                    arcs=active_arcs,
                    start=int(k),
                    first_departure_arcs=w_arcs,
                    successor_links=b_links,
                )
                walk_eval = walk_model if not unused_model else walk_euler
                truck_assigned = [int(i) for i in demand_nodes if _var_val(c[s, k, i], use_pool) > 0.5]
                drone_assigned = [int(i) for i in demand_nodes if _var_val(z[s, k, i], use_pool) > 0.5]
                drone_seq = [int(i) for i in drone_orders[(s, k)] if _var_val(z[s, k, i], use_pool) > 0.5]
                drone_arrival = {str(i): _var_val(p[s, i], use_pool) for i in drone_seq}
                scen_rec["depots"].append(
                    {
                        "depot_internal": int(k),
                        "truck_assigned_demands": truck_assigned,
                        "drone_assigned_demands": drone_assigned,
                        "truck_active_arcs": [[int(i), int(j)] for (i, j) in active_arcs],
                        # Legacy Euler walk for backward compatibility.
                        "truck_walk": [int(v) for v in walk_euler],
                        "truck_walk_euler": [int(v) for v in walk_euler],
                        "truck_unused_arcs_after_walk": [[int(i), int(j)] for (i, j) in unused_euler],
                        # New route-time semantics fields.
                        "truck_walk_model_order": [int(v) for v in walk_model],
                        "truck_unused_arcs_after_model_order": [[int(i), int(j)] for (i, j) in unused_model],
                        "truck_walk_eval_preferred": [int(v) for v in walk_eval],
                        "truck_first_departure_arcs_w": [[int(i), int(j)] for (i, j) in w_arcs],
                        "truck_last_departure_arcs_a": [[int(i), int(j)] for (i, j) in a_arcs],
                        "truck_successor_links_b": [[int(i), int(j), int(l)] for (i, j, l) in b_links],
                        "drone_sequence_spt": drone_seq,
                        "drone_arrival_time": drone_arrival,
                    }
                )
            scenario_routes.append(scen_rec)
        return {"scenario_routes": scenario_routes}

    def _extract_debug_variable_dump(use_pool: bool = False) -> Dict[str, object]:
        f_pos = []
        for (s, i, j) in f_keys:
            val = _var_val(f[s, i, j], use_pool)
            if abs(val) > 1e-9:
                f_pos.append([int(s), int(i), int(j), float(val)])

        p_pos = []
        phi_pos = []
        psi_pos = []
        sigma_pos = []
        for s in range(num_s):
            for i in demand_nodes:
                p_val = _var_val(p[s, i], use_pool)
                phi_val = _var_val(phi[s, i], use_pool)
                psi_val = _var_val(psi[s, i], use_pool)
                sigma_val = _var_val(sigma[s, i], use_pool)
                p_pos.append([int(s), int(i), float(p_val)])
                phi_pos.append([int(s), int(i), float(phi_val)])
                psi_pos.append([int(s), int(i), float(psi_val)])
                sigma_pos.append([int(s), int(i), float(sigma_val)])

        u_pos = []
        h_pos = []
        c_pos = []
        z_pos = []
        for s in range(num_s):
            for k in depots:
                for i in demand_nodes:
                    u_val = _var_val(u_vis[s, k, i], use_pool)
                    h_val = _var_val(h[s, k, i], use_pool)
                    c_val = _var_val(c[s, k, i], use_pool)
                    z_val = _var_val(z[s, k, i], use_pool)
                    if abs(u_val) > 1e-9:
                        u_pos.append([int(s), int(k), int(i), float(u_val)])
                    if abs(h_val) > 1e-9:
                        h_pos.append([int(s), int(k), int(i), float(h_val)])
                    if c_val > 0.5:
                        c_pos.append([int(s), int(k), int(i), float(c_val)])
                    if z_val > 0.5:
                        z_pos.append([int(s), int(k), int(i), float(z_val)])

        y_pos = []
        for (s, k, i, j) in y_keys:
            val = _var_val(y[s, k, i, j], use_pool)
            if val > 0.5:
                y_pos.append([int(s), int(k), int(i), int(j), float(val)])

        w_pos = []
        for (s, k, i, j) in w_keys:
            val = _var_val(w[s, k, i, j], use_pool)
            if val > 0.5:
                w_pos.append([int(s), int(k), int(i), int(j), float(val)])

        a_pos = []
        for (s, k, i, j) in a_keys:
            val = _var_val(a_last[s, k, i, j], use_pool)
            if val > 0.5:
                a_pos.append([int(s), int(k), int(i), int(j), float(val)])

        b_pos = []
        for (s, i, j, l) in b_keys:
            val = _var_val(b[s, i, j, l], use_pool)
            if val > 0.5:
                b_pos.append([int(s), int(i), int(j), int(l), float(val)])

        return {
            "f_positive": f_pos,
            "p_all": p_pos,
            "phi_all": phi_pos,
            "psi_all": psi_pos,
            "sigma_all": sigma_pos,
            "u_positive": u_pos,
            "h_positive": h_pos,
            "c_positive": c_pos,
            "z_positive": z_pos,
            "y_positive": y_pos,
            "w_positive": w_pos,
            "a_positive": a_pos,
            "b_positive": b_pos,
        }

    result = {
        "status": int(model.Status),
        "status_name": _status_name(int(model.Status)),
        "objective": None,
        "best_bound": None,
        "achieved_mip_gap": None,
        "achieved_mip_gap_pct": None,
        "runtime_sec": float(model.Runtime),
        "model_stats": {
            "big_m_time": float(m_time),
            "truck_horizon": float(truck_horizon),
            "drone_horizon": float(drone_horizon),
            "truck_model": "paper_exact",
            "var_counts": {
                "c": num_s * len(depots) * len(demand_nodes),
                "z": num_s * len(depots) * len(demand_nodes),
                "y": len(y_keys),
                "w": len(w_keys),
                "a_last": len(a_keys),
                "b": len(b_keys),
                "f": len(f_keys),
                "p": num_s * len(demand_nodes),
                "phi": num_s * len(demand_nodes),
                "psi": num_s * len(demand_nodes),
                "sigma": num_s * len(demand_nodes),
                "u_vis": num_s * len(depots) * len(demand_nodes),
                "h": num_s * len(depots) * len(demand_nodes),
            },
        },
    }
    if model.SolCount > 0:
        result["objective"] = float(model.ObjVal)
        result["best_bound"] = float(model.ObjBound)
        if abs(model.ObjVal) > 1e-9:
            result["achieved_mip_gap"] = abs(model.ObjVal - model.ObjBound) / abs(model.ObjVal)
        else:
            result["achieved_mip_gap"] = 0.0
        result["achieved_mip_gap_pct"] = 100.0 * result["achieved_mip_gap"]
        truck_flow_val = float(truck_flow_component.getValue())
        truck_arc_val = float(truck_arc_component.getValue())
        drone_val = float(drone_component.getValue())
        obj_sum_val = float(truck_flow_val - truck_arc_val + drone_val)
        result["objective_components_incumbent"] = {
            "truck_flow_component": truck_flow_val,
            "truck_arc_component": truck_arc_val,
            "drone_component": drone_val,
            "flow_minus_arc_plus_drone": obj_sum_val,
            "normalization_coeff": float(coeff),
            "normalized_objective_from_components": float(coeff * obj_sum_val),
        }

        if return_solution_details:
            result["solution_details"] = _extract_solution_details(solution_number=0, use_pool=False)
        if return_debug_variable_dump:
            result["debug_variable_dump"] = _extract_debug_variable_dump(use_pool=False)

        if bool(arrival_rescore):
            max_pool = max(1, int(arrival_pool_solutions))
            pool_count = int(model.SolCount)
            eval_count = min(pool_count, max_pool)
            candidate_metrics: List[Dict[str, object]] = []
            best_rec: Optional[Dict[str, object]] = None
            best_details: Optional[Dict] = None

            for sol_no in range(eval_count):
                details = _extract_solution_details(solution_number=sol_no, use_pool=True)
                arr_eval = _arrival_sim_from_solution_routes(
                    scenarios=scenarios,
                    scenario_routes=details.get("scenario_routes", []),
                    num_open_depots=len(depots),
                )
                try:
                    model.Params.SolutionNumber = int(sol_no)
                    paper_obj_sol = float(model.PoolObjVal)
                except Exception:
                    paper_obj_sol = None

                rec = {
                    "solution_number": int(sol_no),
                    "paper_objective": paper_obj_sol,
                    "arrival_sim_objective": arr_eval.get("weighted_avg"),
                    "arrival_sim_scenario_values": arr_eval.get("scenario_values", []),
                }
                candidate_metrics.append(rec)

                cur_arr = rec.get("arrival_sim_objective")
                if cur_arr is None:
                    continue
                if best_rec is None or float(cur_arr) < float(best_rec.get("arrival_sim_objective")):
                    best_rec = rec
                    best_details = details

            result["arrival_rescore"] = {
                "enabled": True,
                "pool_solution_count": int(pool_count),
                "evaluated_solution_count": int(eval_count),
                "paper_objective_incumbent": float(result["objective"]) if result.get("objective") is not None else None,
                "arrival_sim_objective_incumbent": (
                    next(
                        (
                            float(r.get("arrival_sim_objective"))
                            for r in candidate_metrics
                            if int(r.get("solution_number", -1)) == 0 and r.get("arrival_sim_objective") is not None
                        ),
                        None,
                    )
                ),
                "candidate_metrics": candidate_metrics,
                "best_solution_number": (None if best_rec is None else int(best_rec.get("solution_number", 0))),
                "best_arrival_objective": (None if best_rec is None else float(best_rec.get("arrival_sim_objective"))),
                "best_solution_paper_objective": (None if best_rec is None else best_rec.get("paper_objective")),
            }
            if return_solution_details and best_details is not None:
                result["arrival_rescore"]["best_solution_details"] = best_details
    elif int(model.Status) == GRB.INFEASIBLE:
        try:
            model.computeIIS()
            iis_names = [c.ConstrName for c in model.getConstrs() if c.IISConstr]
            result["iis_top"] = iis_names[:40]
            result["iis_count"] = len(iis_names)
        except Exception as exc:
            result["iis_error"] = str(exc)

    return result


def run_instance(
    instance_name: str,
    instances_root: str,
    p: int,
    time_limit: float,
    mip_gap: float,
    mip_gap_abs: Optional[float],
    threads: int,
    max_scenarios: int,
    max_demands: int,
    output_flag: int,
    mip_focus: int,
    heuristics: float,
    presolve: int,
    cuts: int,
    symmetry: int,
    arrival_rescore: bool = False,
    arrival_pool_solutions: int = 16,
    arrival_pool_search_mode: int = 2,
    arrival_pool_gap: Optional[float] = None,
    select_by_arrival: bool = False,
    continue_on_error: bool = False,
) -> Dict:
    if bool(select_by_arrival) and not bool(arrival_rescore):
        raise ValueError("select_by_arrival=True requires arrival_rescore=True.")

    base, scenarios, demand_nodes, candidates = _prepare_data(
        instance_name, instances_root, max_scenarios, max_demands
    )
    if p <= 0 or p > len(candidates):
        raise ValueError(f"p must be in [1, {len(candidates)}], got {p}")

    depot_base_map = dict(getattr(base, "depot_base_map", {}))
    rows = []
    for combo in combinations(candidates, p):
        open_depots = sorted(int(k) for k in combo)
        row = {
            "open_depots": open_depots,
            "open_depots_base": _map_x_to_base(open_depots, depot_base_map),
            "open_depots_idx": _map_x_to_idx(open_depots, candidates),
            "is_best": 0,
            "rank_by_objective": None,
            "error": "",
        }
        try:
            solved = solve_fixed_combo_commodity(
                instance_name=instance_name,
                scenarios=scenarios,
                demand_nodes=demand_nodes,
                open_depots=open_depots,
                time_limit=time_limit,
                mip_gap=mip_gap,
                mip_gap_abs=mip_gap_abs,
                threads=threads,
                output_flag=output_flag,
                mip_focus=mip_focus,
                heuristics=heuristics,
                presolve=presolve,
                cuts=cuts,
                symmetry=symmetry,
                return_solution_details=True,
                arrival_rescore=bool(arrival_rescore),
                arrival_pool_solutions=int(arrival_pool_solutions),
                arrival_pool_search_mode=int(arrival_pool_search_mode),
                arrival_pool_gap=arrival_pool_gap,
            )
            row.update(solved)
        except Exception as exc:
            if not continue_on_error:
                raise
            row.update(
                {
                    "status": None,
                    "status_name": "ERROR",
                    "objective": None,
                    "best_bound": None,
                    "achieved_mip_gap": None,
                    "achieved_mip_gap_pct": None,
                    "runtime_sec": 0.0,
                }
            )
            row["error"] = str(exc)
        rows.append(row)

    rows_by_paper = sorted(
        rows,
        key=lambda r: float(r["objective"]) if r.get("objective") is not None else float("inf"),
    )
    for rank, row in enumerate(rows_by_paper, start=1):
        row["rank_by_objective"] = int(rank)

    def _arrival_key(r: Dict) -> float:
        v = r.get("arrival_rescore", {}).get("best_arrival_objective")
        return float(v) if v is not None else float("inf")

    rows_by_arrival = sorted(rows, key=_arrival_key)
    for rank, row in enumerate(rows_by_arrival, start=1):
        row["rank_by_arrival_rescore"] = int(rank)

    if bool(select_by_arrival):
        rows_sorted = rows_by_arrival
        selection_metric = "arrival_rescore"
    else:
        rows_sorted = rows_by_paper
        selection_metric = "paper_objective"

    feasible = [r for r in rows_sorted if r.get("objective") is not None]
    best = feasible[0] if feasible else None
    if best is not None:
        best["is_best"] = 1

    summary = {
        "instance": instance_name,
        "num_combinations": len(rows_sorted),
        "feasible_count": len(feasible),
        "optimal_count": sum(1 for r in rows_sorted if r.get("status_name") == "OPTIMAL"),
        "time_limit_count": sum(1 for r in rows_sorted if r.get("status_name") == "TIME_LIMIT"),
        "error_count": sum(1 for r in rows_sorted if r.get("status_name") == "ERROR"),
        "avg_runtime_sec": statistics.mean([float(r.get("runtime_sec", 0.0)) for r in rows_sorted]) if rows_sorted else None,
        "selection_metric": selection_metric,
        "best_objective": float(best["objective"]) if best else None,
        "best_arrival_rescore_objective": (
            float(best.get("arrival_rescore", {}).get("best_arrival_objective"))
            if best and best.get("arrival_rescore", {}).get("best_arrival_objective") is not None
            else None
        ),
        "best_open_depots": best["open_depots"] if best else [],
        "best_open_depots_base": best["open_depots_base"] if best else [],
        "best_open_depots_idx": best["open_depots_idx"] if best else [],
    }

    return {
        "instance": instance_name,
        "p": int(p),
        "num_demands": len(demand_nodes),
        "num_scenarios": len(scenarios),
        "candidate_depots": list(candidates),
        "depot_base_map": depot_base_map,
        "solver_params": {
            "time_limit_sec_per_combination": float(time_limit),
            "target_mip_gap": float(mip_gap),
            "target_mip_gap_abs": None if mip_gap_abs is None or mip_gap_abs < 0 else float(mip_gap_abs),
            "threads": int(threads),
            "truck_model": "paper_exact",
            "max_scenarios": int(max_scenarios),
            "max_demands": int(max_demands),
            "mip_focus": int(mip_focus),
            "heuristics": float(heuristics),
            "presolve": int(presolve),
            "cuts": int(cuts),
            "symmetry": int(symmetry),
            "drone_order_rule": "SPT_fixed_by_roundtrip_time",
            "arrival_rescore": bool(arrival_rescore),
            "arrival_pool_solutions": int(arrival_pool_solutions),
            "arrival_pool_search_mode": int(arrival_pool_search_mode),
            "arrival_pool_gap": (None if arrival_pool_gap is None or arrival_pool_gap < 0 else float(arrival_pool_gap)),
            "select_by_arrival": bool(select_by_arrival),
        },
        "summary": summary,
        "combination_results": rows_sorted,
    }


def run_batch(
    instance_names: Sequence[str],
    instances_root: str,
    p: int,
    time_limit: float,
    mip_gap: float,
    mip_gap_abs: Optional[float],
    threads: int,
    max_scenarios: int,
    max_demands: int,
    output_flag: int,
    mip_focus: int,
    heuristics: float,
    presolve: int,
    cuts: int,
    symmetry: int,
    arrival_rescore: bool = False,
    arrival_pool_solutions: int = 16,
    arrival_pool_search_mode: int = 2,
    arrival_pool_gap: Optional[float] = None,
    select_by_arrival: bool = False,
    continue_on_error: bool = False,
) -> Dict:
    runs = []
    for name in instance_names:
        try:
            runs.append(
                run_instance(
                    instance_name=name,
                    instances_root=instances_root,
                    p=p,
                    time_limit=time_limit,
                    mip_gap=mip_gap,
                    mip_gap_abs=mip_gap_abs,
                    threads=threads,
                    max_scenarios=max_scenarios,
                    max_demands=max_demands,
                    output_flag=output_flag,
                    mip_focus=mip_focus,
                    heuristics=heuristics,
                    presolve=presolve,
                    cuts=cuts,
                    symmetry=symmetry,
                    arrival_rescore=arrival_rescore,
                    arrival_pool_solutions=arrival_pool_solutions,
                    arrival_pool_search_mode=arrival_pool_search_mode,
                    arrival_pool_gap=arrival_pool_gap,
                    select_by_arrival=select_by_arrival,
                    continue_on_error=continue_on_error,
                )
            )
        except Exception as exc:
            if not continue_on_error:
                raise
            runs.append(
                {
                    "instance": name,
                    "summary": {
                        "instance": name,
                        "num_combinations": 0,
                        "feasible_count": 0,
                        "optimal_count": 0,
                        "time_limit_count": 0,
                        "error_count": 1,
                        "avg_runtime_sec": 0.0,
                        "selection_metric": ("arrival_rescore" if bool(select_by_arrival) else "paper_objective"),
                        "best_objective": None,
                        "best_arrival_rescore_objective": None,
                        "best_open_depots": [],
                        "best_open_depots_base": [],
                        "best_open_depots_idx": [],
                    },
                    "error": str(exc),
                    "combination_results": [],
                }
            )

    best_values = [
        float(r["summary"]["best_objective"])
        for r in runs
        if r.get("summary", {}).get("best_objective") is not None
    ]
    summary = {
        "instance_count": len(runs),
        "solved_count": sum(1 for r in runs if r.get("summary", {}).get("feasible_count", 0) > 0),
        "error_count": sum(1 for r in runs if r.get("summary", {}).get("error_count", 0) > 0),
        "avg_best_objective": statistics.mean(best_values) if best_values else None,
    }
    return {"runs": runs, "summary": summary}


def save_combo_csv(path: str, payload: Dict) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "instance",
                "is_best",
                "rank_by_objective",
                "open_depots",
                "open_depots_base",
                "open_depots_idx",
                "status_name",
                "objective",
                "best_bound",
                "achieved_mip_gap",
                "achieved_mip_gap_pct",
                "rank_by_arrival_rescore",
                "arrival_best_objective",
                "arrival_best_solution_number",
                "arrival_pool_solution_count",
                "runtime_sec",
                "error",
            ]
        )
        runs = payload["runs"] if "runs" in payload else [payload]
        for run in runs:
            for row in run.get("combination_results", []):
                w.writerow(
                    [
                        run.get("instance", ""),
                        int(row.get("is_best", 0)),
                        row.get("rank_by_objective"),
                        row.get("open_depots", []),
                        row.get("open_depots_base", []),
                        row.get("open_depots_idx", []),
                        row.get("status_name", ""),
                        row.get("objective"),
                        row.get("best_bound"),
                        row.get("achieved_mip_gap"),
                        row.get("achieved_mip_gap_pct"),
                        row.get("rank_by_arrival_rescore"),
                        (row.get("arrival_rescore", {}) or {}).get("best_arrival_objective"),
                        (row.get("arrival_rescore", {}) or {}).get("best_solution_number"),
                        (row.get("arrival_rescore", {}) or {}).get("pool_solution_count"),
                        row.get("runtime_sec"),
                        row.get("error", ""),
                    ]
                )


def save_best_csv(path: str, payload: Dict) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "instance",
                "best_objective",
                "best_arrival_rescore_objective",
                "selection_metric",
                "best_open_depots",
                "best_open_depots_base",
                "best_open_depots_idx",
                "feasible_count",
                "optimal_count",
                "time_limit_count",
                "error_count",
            ]
        )
        runs = payload["runs"] if "runs" in payload else [payload]
        for run in runs:
            s = run.get("summary", {})
            w.writerow(
                [
                    run.get("instance", ""),
                    s.get("best_objective"),
                    s.get("best_arrival_rescore_objective"),
                    s.get("selection_metric"),
                    s.get("best_open_depots", []),
                    s.get("best_open_depots_base", []),
                    s.get("best_open_depots_idx", []),
                    s.get("feasible_count"),
                    s.get("optimal_count"),
                    s.get("time_limit_count"),
                    s.get("error_count"),
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enumerate depot combinations and solve each with Gurobi routing model."
    )
    parser.add_argument("--instances-root", default="./Instance")
    parser.add_argument("--instance", default="Instance3", help="Single instance name, e.g. Instance3")
    parser.add_argument(
        "--instances",
        default="2,5",
        help="Batch selector, e.g. 1-10 | 1,3,7 | Instance2,Instance5. Overrides --instance.",
    )
    parser.add_argument("--p", type=int, default=2, help="Number of depots to open.")
    parser.add_argument("--time-limit", type=float, default=3600.0, help="Per-combination time limit (sec).")
    parser.add_argument("--mip-gap", type=float, default=0.0, help="Target relative MIP gap.")
    parser.add_argument("--mip-gap-abs", type=float, default=-1.0, help="Target absolute MIP gap; <0 disables.")
    parser.add_argument("--threads", type=int, default=0, help="0 means Gurobi default.")
    parser.add_argument("--max-scenarios", type=int, default=0, help="0 means all scenarios.")
    parser.add_argument("--max-demands", type=int, default=0, help="0 means all demand nodes.")
    parser.add_argument("--mip-focus", type=int, default=-1, help="Gurobi MIPFocus in {0,1,2,3}; -1 keeps default.")
    parser.add_argument("--heuristics", type=float, default=-1.0, help="Gurobi Heuristics in [0,1]; negative keeps default.")
    parser.add_argument("--presolve", type=int, default=-1, help="Gurobi Presolve in {-1,0,1,2}.")
    parser.add_argument("--cuts", type=int, default=-1, help="Gurobi Cuts in {-1,0,1,2,3}.")
    parser.add_argument("--symmetry", type=int, default=-1, help="Gurobi Symmetry in {-1,0,1,2}.")
    parser.add_argument("--arrival-rescore", action="store_true", help="Enable solution-pool arrival_sim rescoring.")
    parser.add_argument("--arrival-pool-solutions", type=int, default=16, help="Max number of pool solutions to evaluate by arrival_sim.")
    parser.add_argument("--arrival-pool-search-mode", type=int, default=2, help="Gurobi PoolSearchMode in {0,1,2} when arrival-rescore is on.")
    parser.add_argument("--arrival-pool-gap", type=float, default=-1.0, help="Gurobi PoolGap; <0 disables.")
    parser.add_argument("--select-by-arrival", action="store_true", help="Select best depot combo by arrival_sim rescored value.")
    parser.add_argument("--out", default="outputs/gurobi_exact_small_enumeration.json")
    parser.add_argument("--out-csv", default="", help="Optional CSV with one row per combination.")
    parser.add_argument("--out-best-csv", default="", help="Optional CSV with one row per instance (best only).")
    parser.add_argument("--per-instance-dir", default="", help="Optional directory to save per-instance JSON files.")
    parser.add_argument("--quiet", action="store_true", help="Silence Gurobi log output.")
    parser.add_argument("--continue-on-error", action="store_true", help="In batch mode, skip failed instances.")
    args = parser.parse_args()

    if bool(args.select_by_arrival) and not bool(args.arrival_rescore):
        raise ValueError("--select-by-arrival requires --arrival-rescore.")

    instance_names = parse_instance_names(args.instances) if args.instances.strip() else [args.instance]

    batch = run_batch(
        instance_names=instance_names,
        instances_root=args.instances_root,
        p=args.p,
        time_limit=args.time_limit,
        mip_gap=args.mip_gap,
        mip_gap_abs=(None if args.mip_gap_abs < 0 else args.mip_gap_abs),
        threads=args.threads,
        max_scenarios=args.max_scenarios,
        max_demands=args.max_demands,
        output_flag=(0 if args.quiet else 1),
        mip_focus=args.mip_focus,
        heuristics=args.heuristics,
        presolve=args.presolve,
        cuts=args.cuts,
        symmetry=args.symmetry,
        arrival_rescore=bool(args.arrival_rescore),
        arrival_pool_solutions=int(args.arrival_pool_solutions),
        arrival_pool_search_mode=int(args.arrival_pool_search_mode),
        arrival_pool_gap=(None if args.arrival_pool_gap < 0 else args.arrival_pool_gap),
        select_by_arrival=bool(args.select_by_arrival),
        continue_on_error=bool(args.continue_on_error),
    )

    payload = batch if len(instance_names) > 1 else batch["runs"][0]

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Saved JSON: {args.out}")

    if args.out_csv:
        save_combo_csv(args.out_csv, payload)
        print(f"Saved CSV: {args.out_csv}")
    if args.out_best_csv:
        save_best_csv(args.out_best_csv, payload)
        print(f"Saved best CSV: {args.out_best_csv}")

    if args.per_instance_dir:
        os.makedirs(args.per_instance_dir, exist_ok=True)
        for run in batch["runs"]:
            pth = os.path.join(args.per_instance_dir, f"{run['instance']}.json")
            with open(pth, "w", encoding="utf-8") as f:
                json.dump(run, f, ensure_ascii=False, indent=2)
        print(f"Saved per-instance JSON files to: {args.per_instance_dir}")

    print("Summary:", batch["summary"])
    for run in batch["runs"]:
        s = run.get("summary", {})
        print(
            f"{run.get('instance')}: best_obj={s.get('best_objective')} "
            f"best_arrival={s.get('best_arrival_rescore_objective')} "
            f"selection_metric={s.get('selection_metric')} "
            f"best_idx={s.get('best_open_depots_idx')} "
            f"optimal_combo_count={s.get('optimal_count')} "
            f"time_limit_combo_count={s.get('time_limit_count')}"
        )


if __name__ == "__main__":
    main()
