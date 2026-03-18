"""
Exact two-stage stochastic MILP for small-sized instances using Gurobi.

This script is intended for the paper's small-sized benchmark style:
- 15 demand nodes
- 5 candidate depots
- 10 scenarios

Modeling notes:
1) First-stage decision:
   - x[k] in {0,1}: open depot k
   - sum_k x[k] = p

2) Second-stage (for each scenario s):
   - Truck assignment/routing from each open depot (single route with sink).
   - Drone assignment and sequence timing (single drone per open depot).
   - Each demand served exactly once (truck or drone).
   - Arrival time variable a[s,i] for each demand i.

3) Objective:
   Minimize expected average arrival time:
       (1 / (|S| * |D|)) * sum_{s in S} sum_{i in D} a[s,i]

Important:
- This is an exact MIP and can be very slow.
- It supports both single-instance and batch execution from CLI.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
from collections import deque
from typing import Dict, List, Optional, Sequence, Set, Tuple

import alg
from main import load_instance_data
from paper_eval_common import parse_instance_names

try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "gurobipy is required to run this script. Install Gurobi and gurobipy first."
    ) from exc


def _max_finite_time(scenarios, demand_nodes: List[int], depots: List[int]) -> float:
    max_t = 1.0
    all_nodes = list(dict.fromkeys(demand_nodes + depots))
    for scen in scenarios:
        for i in all_nodes:
            for j in all_nodes:
                t = scen.truck_times.get(i, {}).get(j, math.inf)
                if not math.isinf(t):
                    max_t = max(max_t, float(t))
                v = scen.drone_times.get(i, {}).get(j, math.inf)
                if not math.isinf(v):
                    max_t = max(max_t, float(v))
    return max_t


def _drone_reachable(scen, depot: int, demand: int) -> bool:
    reachable = scen.drone_reachability.get(depot, [])
    if not reachable:
        # Fallback if reachability list is absent.
        return not math.isinf(scen.drone_times.get(depot, {}).get(demand, math.inf))
    return demand in set(reachable)


def _truck_arc_feasible(scen, i: int, j: int) -> bool:
    if i == j:
        return False
    t = scen.truck_times.get(i, {}).get(j, math.inf)
    return not math.isinf(t)


def _truck_reachable_from_depot(scen, depot: int, demand_nodes: List[int]) -> Set[int]:
    """
    Directed reachability from depot over nodes {depot} U demand_nodes using finite truck arcs.
    Used only as a safe pruning rule:
    if demand i is not reachable from depot k, then u[s,k,i] must be 0.
    """
    node_set = [depot] + list(demand_nodes)
    node_membership = set(node_set)
    visited: Set[int] = {depot}
    q: deque = deque([depot])
    while q:
        u = q.popleft()
        row = scen.truck_times.get(u, {})
        for v in node_set:
            if v in visited or v not in node_membership:
                continue
            t = row.get(v, math.inf)
            if not math.isinf(t):
                visited.add(v)
                q.append(v)
    return {i for i in demand_nodes if i in visited}


def _time_horizons(
    scenarios,
    demand_nodes: List[int],
    depots: List[int],
    drone_reachable: Dict[Tuple[int, int], Set[int]],
) -> Tuple[float, float, float]:
    """
    Compute tighter global horizons:
    - max_arc_time: max finite truck/drone arc over relevant nodes
    - truck_horizon: conservative upper bound on truck arrival timeline
    - drone_horizon: conservative upper bound on drone arrival timeline
    """
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
                v = scen.drone_times.get(k, {}).get(i, math.inf)
                if not math.isinf(v):
                    max_drone_rt = max(max_drone_rt, float(v))
    max_arc_time = max(max_truck_arc, max_drone_rt)
    # Each demand can at most add one arc-like segment on the route/schedule.
    truck_horizon = n * max_truck_arc
    drone_horizon = n * max_drone_rt
    return float(max_arc_time), float(truck_horizon), float(drone_horizon)


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


def build_and_solve(
    instance_name: str,
    instances_root: str,
    num_open_depots: int,
    time_limit: float,
    mip_gap: float,
    mip_gap_abs: Optional[float],
    threads: int,
    max_scenarios: int = 0,
    max_demands: int = 0,
    output_flag: int = 1,
    mip_focus: int = -1,
    heuristics: float = -1.0,
    presolve: int = -1,
    cuts: int = -1,
    symmetry: int = -1,
    warm_start: bool = False,
    warm_start_seed: int = 123,
    warm_start_use_cutoff: bool = False,
) -> Dict:
    folder = os.path.join(instances_root, instance_name)
    if not os.path.exists(folder):
        raise FileNotFoundError(f"Instance folder not found: {folder}")

    base_instance, scenarios = load_instance_data(folder)
    if not scenarios:
        raise ValueError("No scenarios loaded from t.txt.")
    depot_base_map = dict(getattr(base_instance, "depot_base_map", {}))

    demand_nodes = list(base_instance.demand_nodes)
    depots = list(base_instance.candidate_depots)
    if max_demands > 0:
        demand_nodes = demand_nodes[: max_demands]
    if max_scenarios > 0:
        scenarios = scenarios[: max_scenarios]
    num_s = len(scenarios)

    # Keep only selected demand nodes in scenario metadata.
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

    if num_open_depots <= 0 or num_open_depots > len(depots):
        raise ValueError(f"p must be in [1, {len(depots)}], got {num_open_depots}")

    # Precompute safe pruning sets.
    truck_reachable = {
        (s, k): _truck_reachable_from_depot(scen, k, demand_nodes)
        for s, scen in enumerate(scenarios)
        for k in depots
    }
    drone_reachable = {
        (s, k): {i for i in demand_nodes if _drone_reachable(scen, k, i)}
        for s, scen in enumerate(scenarios)
        for k in depots
    }

    # Tighter Big-M values from data horizons.
    max_arc_time, truck_horizon, drone_horizon = _time_horizons(
        scenarios, demand_nodes, depots, drone_reachable
    )
    # M_time is used on constraints like a_i >= a_p + t - M(1-y)
    # so M should dominate (max possible timeline + one arc).
    M_time = max(truck_horizon, drone_horizon) + max_arc_time
    M_drone = drone_horizon + max_arc_time
    M_order = float(len(demand_nodes))

    model = gp.Model(f"exact_small_{instance_name}")
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

    # -----------------------------
    # First-stage variables
    # -----------------------------
    x = model.addVars(depots, vtype=GRB.BINARY, name="x")
    for k in depots:
        x[k].BranchPriority = 20
    model.addConstr(gp.quicksum(x[k] for k in depots) == num_open_depots, name="open_p_depots")

    # -----------------------------
    # Second-stage variables
    # -----------------------------
    # a[s, i]: arrival time of demand i in scenario s.
    a = model.addVars(range(num_s), demand_nodes, lb=0.0, vtype=GRB.CONTINUOUS, name="a")

    # u[s, k, i]: demand i served by truck route of depot k in scenario s.
    u = model.addVars(range(num_s), depots, demand_nodes, vtype=GRB.BINARY, name="u")

    # z[s, k, i]: demand i served by drone from depot k in scenario s.
    z = model.addVars(range(num_s), depots, demand_nodes, vtype=GRB.BINARY, name="z")

    # Truck route arcs with sink node per depot.
    sink = {k: f"sink_{k}" for k in depots}
    y_keys: List[Tuple[int, int, object, object]] = []
    for s, scen in enumerate(scenarios):
        for k in depots:
            reach_k = sorted(truck_reachable[(s, k)])
            from_nodes = [k] + reach_k
            to_nodes = reach_k + [sink[k]]
            for i in from_nodes:
                for j in to_nodes:
                    if j == sink[k]:
                        # Allow arc to sink from depot or demand (finish route).
                        y_keys.append((s, k, i, j))
                    else:
                        if _truck_arc_feasible(scen, i, j):
                            y_keys.append((s, k, i, j))

    y = model.addVars(y_keys, vtype=GRB.BINARY, name="y")

    # MTZ order variables for truck subtour elimination on demand nodes.
    ord_t = model.addVars(
        range(num_s), depots, demand_nodes, lb=0.0, ub=M_order, vtype=GRB.CONTINUOUS, name="ord_t"
    )

    # Drone timing variables.
    d_start = model.addVars(
        range(num_s), depots, demand_nodes, lb=0.0, vtype=GRB.CONTINUOUS, name="d_start"
    )

    # Pairwise precedence binary for drone sequencing.
    b_keys = []
    for s, scen in enumerate(scenarios):
        for k in depots:
            reach_drone = sorted(drone_reachable[(s, k)])
            for idx_i, i in enumerate(reach_drone):
                for j in reach_drone[idx_i + 1 :]:
                    b_keys.append((s, k, i, j))
    b = model.addVars(b_keys, vtype=GRB.BINARY, name="b")

    # -----------------------------
    # Constraints
    # -----------------------------
    for s, scen in enumerate(scenarios):
        # Coverage: each demand served exactly once.
        for i in demand_nodes:
            model.addConstr(
                gp.quicksum(u[s, k, i] + z[s, k, i] for k in depots) == 1,
                name=f"cover_s{s}_i{i}",
            )

        for k in depots:
            reach_truck = truck_reachable[(s, k)]
            reach_drone = drone_reachable[(s, k)]

            # Depot-open linking.
            for i in demand_nodes:
                model.addConstr(u[s, k, i] <= x[k], name=f"u_open_s{s}_k{k}_i{i}")
                model.addConstr(z[s, k, i] <= x[k], name=f"z_open_s{s}_k{k}_i{i}")
                if i not in reach_truck:
                    model.addConstr(u[s, k, i] == 0, name=f"u_reach_s{s}_k{k}_i{i}")
                if i not in reach_drone:
                    model.addConstr(z[s, k, i] == 0, name=f"z_reach_s{s}_k{k}_i{i}")

            # Truck route start/end at sink flow.
            out_from_depot = gp.quicksum(
                y[s, k, k, j]
                for j in demand_nodes + [sink[k]]
                if (s, k, k, j) in y
            )
            into_sink = gp.quicksum(
                y[s, k, i, sink[k]]
                for i in [k] + demand_nodes
                if (s, k, i, sink[k]) in y
            )
            model.addConstr(out_from_depot == x[k], name=f"start_s{s}_k{k}")
            model.addConstr(into_sink == x[k], name=f"end_s{s}_k{k}")

            # Flow conservation for truck-served demand.
            for i in demand_nodes:
                in_i = gp.quicksum(
                    y[s, k, pred, i]
                    for pred in [k] + demand_nodes
                    if (s, k, pred, i) in y
                )
                out_i = gp.quicksum(
                    y[s, k, i, q]
                    for q in demand_nodes + [sink[k]]
                    if (s, k, i, q) in y
                )
                model.addConstr(in_i == u[s, k, i], name=f"truck_in_s{s}_k{k}_i{i}")
                model.addConstr(out_i == u[s, k, i], name=f"truck_out_s{s}_k{k}_i{i}")

                # MTZ link.
                model.addConstr(ord_t[s, k, i] <= M_order * u[s, k, i], name=f"ord_ub_s{s}_k{k}_i{i}")
                model.addConstr(ord_t[s, k, i] >= u[s, k, i], name=f"ord_lb_s{s}_k{k}_i{i}")

            # MTZ subtour elimination on demand-demand arcs.
            for i in demand_nodes:
                for j in demand_nodes:
                    if i == j:
                        continue
                    if (s, k, i, j) not in y:
                        continue
                    model.addConstr(
                        ord_t[s, k, i] - ord_t[s, k, j] + M_order * y[s, k, i, j] <= M_order - 1,
                        name=f"mtz_s{s}_k{k}_i{i}_j{j}",
                    )

            # Truck arrival propagation (if truck arc is used).
            for i in demand_nodes:
                if (s, k, k, i) in y:
                    tki = scen.truck_times[k][i]
                    model.addConstr(
                        a[s, i] >= tki - M_time * (1 - y[s, k, k, i]),
                        name=f"arr_from_depot_s{s}_k{k}_i{i}",
                    )
                for pred in demand_nodes:
                    if (s, k, pred, i) in y:
                        tpi = scen.truck_times[pred][i]
                        model.addConstr(
                            a[s, i] >= a[s, pred] + tpi - M_time * (1 - y[s, k, pred, i]),
                            name=f"arr_truck_s{s}_k{k}_p{pred}_i{i}",
                        )

            # Drone start-time variable active only when assigned.
            for i in demand_nodes:
                if i in reach_drone:
                    model.addConstr(
                        d_start[s, k, i] <= M_drone * z[s, k, i],
                        name=f"d_active_s{s}_k{k}_i{i}",
                    )
                    rt = scen.drone_times[k][i]
                    ow = rt / 2.0
                    model.addConstr(
                        a[s, i] >= d_start[s, k, i] + ow - M_time * (1 - z[s, k, i]),
                        name=f"arr_drone_s{s}_k{k}_i{i}",
                    )
                else:
                    model.addConstr(d_start[s, k, i] == 0.0, name=f"d_zero_s{s}_k{k}_i{i}")

            # Pairwise disjunctive sequencing for single drone at depot k.
            # If both i and j are assigned to this drone, one must be before the other.
            reach_drone_sorted = sorted(reach_drone)
            for idx_i, i in enumerate(reach_drone_sorted):
                for j in reach_drone_sorted[idx_i + 1 :]:
                    bvar = b[s, k, i, j]
                    rt_i = scen.drone_times[k][i]
                    rt_j = scen.drone_times[k][j]

                    # If i and j are both assigned:
                    #   b=1 => i before j
                    #   b=0 => j before i
                    # Extra activation term with z's disables constraints otherwise.
                    model.addConstr(
                        d_start[s, k, j]
                        >= d_start[s, k, i]
                        + rt_i
                        - M_time * (1 - bvar)
                        - M_time * (2 - z[s, k, i] - z[s, k, j]),
                        name=f"dr_seq1_s{s}_k{k}_i{i}_j{j}",
                    )
                    model.addConstr(
                        d_start[s, k, i]
                        >= d_start[s, k, j]
                        + rt_j
                        - M_time * bvar
                        - M_time * (2 - z[s, k, i] - z[s, k, j]),
                        name=f"dr_seq2_s{s}_k{k}_i{i}_j{j}",
                    )

    # -----------------------------
    # Objective: expected average arrival time
    # -----------------------------
    coeff = 1.0 / (num_s * len(demand_nodes))
    model.setObjective(
        coeff * gp.quicksum(a[s, i] for s in range(num_s) for i in demand_nodes),
        GRB.MINIMIZE,
    )

    warm_start_info = {"used": False, "seed": None, "x_open": [], "expected_obj": None, "error": ""}
    if warm_start:
        warm_start_info["seed"] = int(warm_start_seed)
        try:
            ws_cfg = alg.HeuristicConfig(
                num_depots_to_open=num_open_depots,
                num_scenarios=num_s,
                seed=int(warm_start_seed),
                k_max=5,
                l_max=4,
                i_max=3,
                drone_time_is_roundtrip=True,
                normalize_by_num_demands=True,
            )
            ws_x, ws_obj = alg.algorithm_1_stochastic_lrp(base_instance, cfg=ws_cfg, scenarios=scenarios)
            ws_x = set(int(k) for k in ws_x)
            for k in depots:
                x[k].Start = 1.0 if k in ws_x else 0.0
            warm_start_info["used"] = True
            warm_start_info["x_open"] = sorted(ws_x)
            warm_start_info["expected_obj"] = float(ws_obj)
            if warm_start_use_cutoff and ws_obj is not None and not math.isinf(ws_obj):
                model.Params.Cutoff = float(ws_obj)
        except Exception as exc:
            warm_start_info["error"] = str(exc)

    model.optimize()

    # -----------------------------
    # Extract result
    # -----------------------------
    result = {
        "instance": instance_name,
        "num_demands": len(demand_nodes),
        "num_scenarios": num_s,
        "depot_base_map": depot_base_map,
        "solver_params": {
            "p": num_open_depots,
            "time_limit_sec": float(time_limit),
            "target_mip_gap": float(mip_gap),
            "target_mip_gap_abs": None if mip_gap_abs is None or mip_gap_abs < 0 else float(mip_gap_abs),
            "threads": int(threads),
            "max_scenarios": int(max_scenarios),
            "max_demands": int(max_demands),
            "mip_focus": int(mip_focus),
            "heuristics": float(heuristics),
            "presolve": int(presolve),
            "cuts": int(cuts),
            "symmetry": int(symmetry),
            "warm_start": bool(warm_start),
            "warm_start_seed": int(warm_start_seed),
            "warm_start_use_cutoff": bool(warm_start_use_cutoff),
        },
        "model_stats": {
            "big_m_time": float(M_time),
            "big_m_drone": float(M_drone),
            "truck_horizon": float(truck_horizon),
            "drone_horizon": float(drone_horizon),
            "var_counts": {
                "x": len(depots),
                "a": num_s * len(demand_nodes),
                "u": num_s * len(depots) * len(demand_nodes),
                "z": num_s * len(depots) * len(demand_nodes),
                "y": len(y_keys),
                "ord_t": num_s * len(depots) * len(demand_nodes),
                "d_start": num_s * len(depots) * len(demand_nodes),
                "b": len(b_keys),
            },
        },
        "warm_start_info": warm_start_info,
        "status": int(model.Status),
        "status_name": _status_name(int(model.Status)),
        "objective": None,
        "best_bound": None,
        "achieved_mip_gap": None,
        "achieved_mip_gap_pct": None,
        "runtime_sec": float(model.Runtime),
        "open_depots": [],
        "open_depots_base": [],
    }

    if model.SolCount > 0:
        result["objective"] = float(model.ObjVal)
        result["best_bound"] = float(model.ObjBound)
        if abs(model.ObjVal) > 1e-9:
            result["achieved_mip_gap"] = abs(model.ObjVal - model.ObjBound) / abs(model.ObjVal)
        else:
            result["achieved_mip_gap"] = 0.0
        result["achieved_mip_gap_pct"] = 100.0 * result["achieved_mip_gap"]
        result["open_depots"] = [k for k in depots if x[k].X > 0.5]
        result["open_depots_base"] = [int(depot_base_map.get(k, k)) for k in result["open_depots"]]

    return result


def run_batch(
    instance_names: Sequence[str],
    instances_root: str,
    num_open_depots: int,
    time_limit: float,
    mip_gap: float,
    mip_gap_abs: Optional[float],
    threads: int,
    max_scenarios: int = 0,
    max_demands: int = 0,
    output_flag: int = 1,
    continue_on_error: bool = True,
    mip_focus: int = -1,
    heuristics: float = -1.0,
    presolve: int = -1,
    cuts: int = -1,
    symmetry: int = -1,
    warm_start: bool = False,
    warm_start_seed: int = 123,
    warm_start_use_cutoff: bool = False,
) -> Dict:
    rows = []
    for name in instance_names:
        try:
            row = build_and_solve(
                instance_name=name,
                instances_root=instances_root,
                num_open_depots=num_open_depots,
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
                warm_start=warm_start,
                warm_start_seed=warm_start_seed,
                warm_start_use_cutoff=warm_start_use_cutoff,
            )
            row["error"] = ""
            rows.append(row)
        except Exception as exc:
            if not continue_on_error:
                raise
            rows.append(
                {
                    "instance": name,
                    "status": None,
                    "status_name": "ERROR",
                    "objective": None,
                    "best_bound": None,
                    "achieved_mip_gap": None,
                    "achieved_mip_gap_pct": None,
                    "runtime_sec": 0.0,
                    "open_depots": [],
                    "open_depots_base": [],
                    "error": str(exc),
                }
            )

    feasible = [r for r in rows if r.get("objective") is not None]
    objective_values = [float(r["objective"]) for r in feasible]
    best_row = min(feasible, key=lambda r: float(r["objective"])) if feasible else None

    summary = {
        "instance_count": len(rows),
        "feasible_count": len(feasible),
        "optimal_count": sum(1 for r in rows if r.get("status_name") == "OPTIMAL"),
        "time_limit_count": sum(1 for r in rows if r.get("status_name") == "TIME_LIMIT"),
        "error_count": sum(1 for r in rows if r.get("status_name") == "ERROR"),
        "avg_runtime_sec": statistics.mean([float(r.get("runtime_sec", 0.0)) for r in rows]) if rows else None,
        "avg_objective": statistics.mean(objective_values) if objective_values else None,
        "best_objective": float(best_row["objective"]) if best_row else None,
        "best_instance": best_row["instance"] if best_row else None,
    }
    return {"runs": rows, "summary": summary}


def save_batch_csv(path: str, batch: Dict) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "instance",
                "status_name",
                "objective",
                "best_bound",
                "achieved_mip_gap",
                "achieved_mip_gap_pct",
                "runtime_sec",
                "open_depots",
                "open_depots_base",
                "error",
            ]
        )
        for r in batch.get("runs", []):
            w.writerow(
                [
                    r.get("instance", ""),
                    r.get("status_name", ""),
                    r.get("objective"),
                    r.get("best_bound"),
                    r.get("achieved_mip_gap"),
                    r.get("achieved_mip_gap_pct"),
                    r.get("runtime_sec"),
                    r.get("open_depots", []),
                    r.get("open_depots_base", []),
                    r.get("error", ""),
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exact Gurobi MILP for small-sized stochastic truck-drone instances."
    )
    parser.add_argument("--instances-root", default="./Instance")
    parser.add_argument("--instance", default="Instance3", help="Single instance name, e.g. Instance3")
    parser.add_argument(
        "--instances",
        default="",
        help="Batch instance selector, e.g. 1-10 | 1,3,7 | Instance3,Instance4. Overrides --instance.",
    )
    parser.add_argument("--p", type=int, default=2, help="Number of depots to open.")
    parser.add_argument("--time-limit", type=float, default=3600.0)
    parser.add_argument("--mip-gap", type=float, default=0.0, help="Target relative MIP gap.")
    parser.add_argument("--mip-gap-abs", type=float, default=-1.0, help="Target absolute MIP gap; <0 disables.")
    parser.add_argument("--threads", type=int, default=0, help="0 means Gurobi default.")
    parser.add_argument("--max-scenarios", type=int, default=0, help="0 means all scenarios.")
    parser.add_argument("--max-demands", type=int, default=0, help="0 means all demand nodes.")
    parser.add_argument("--mip-focus", type=int, default=-1, help="Gurobi MIPFocus in {0,1,2,3}; -1 keeps default.")
    parser.add_argument(
        "--heuristics",
        type=float,
        default=-1.0,
        help="Gurobi Heuristics parameter in [0,1]; negative keeps default.",
    )
    parser.add_argument("--presolve", type=int, default=-1, help="Gurobi Presolve in {-1,0,1,2}.")
    parser.add_argument("--cuts", type=int, default=-1, help="Gurobi Cuts in {-1,0,1,2,3}.")
    parser.add_argument("--symmetry", type=int, default=-1, help="Gurobi Symmetry in {-1,0,1,2}.")
    parser.add_argument("--warm-start", action="store_true", help="Use heuristic warm-start for x[k].")
    parser.add_argument("--warm-start-seed", type=int, default=123, help="Seed for heuristic warm-start.")
    parser.add_argument(
        "--warm-start-cutoff",
        action="store_true",
        help="If warm-start is enabled, also set Cutoff to warm-start objective.",
    )
    parser.add_argument("--out", default="outputs/gurobi_exact_small_result.json")
    parser.add_argument("--out-csv", default="", help="Optional CSV summary path for batch/single runs.")
    parser.add_argument("--per-instance-dir", default="", help="Optional dir to save one JSON per instance in batch.")
    parser.add_argument("--quiet", action="store_true", help="Silence Gurobi log output.")
    parser.add_argument("--continue-on-error", action="store_true", help="In batch mode, skip failed instances.")
    args = parser.parse_args()

    instance_names = parse_instance_names(args.instances) if args.instances.strip() else [args.instance]
    batch = run_batch(
        instance_names=instance_names,
        instances_root=args.instances_root,
        num_open_depots=args.p,
        time_limit=args.time_limit,
        mip_gap=args.mip_gap,
        mip_gap_abs=(None if args.mip_gap_abs < 0 else args.mip_gap_abs),
        threads=args.threads,
        max_scenarios=args.max_scenarios,
        max_demands=args.max_demands,
        output_flag=(0 if args.quiet else 1),
        continue_on_error=bool(args.continue_on_error),
        mip_focus=args.mip_focus,
        heuristics=args.heuristics,
        presolve=args.presolve,
        cuts=args.cuts,
        symmetry=args.symmetry,
        warm_start=bool(args.warm_start),
        warm_start_seed=args.warm_start_seed,
        warm_start_use_cutoff=bool(args.warm_start_cutoff),
    )

    payload = batch if len(instance_names) > 1 else batch["runs"][0]

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    if args.out_csv:
        save_batch_csv(args.out_csv, batch)
        print(f"Saved CSV: {args.out_csv}")

    if args.per_instance_dir:
        os.makedirs(args.per_instance_dir, exist_ok=True)
        for row in batch["runs"]:
            pth = os.path.join(args.per_instance_dir, f"{row['instance']}.json")
            with open(pth, "w", encoding="utf-8") as f:
                json.dump(row, f, ensure_ascii=False, indent=2)
        print(f"Saved per-instance JSON files to: {args.per_instance_dir}")

    print(f"Saved JSON: {args.out}")
    print("Summary:", batch["summary"])
    for row in batch["runs"]:
        print(
            f"{row.get('instance')}: status={row.get('status_name')} "
            f"obj={row.get('objective')} bound={row.get('best_bound')} "
            f"gap={row.get('achieved_mip_gap_pct')}% runtime={row.get('runtime_sec')}s "
            f"open_base={row.get('open_depots_base')}"
        )


if __name__ == "__main__":
    main()
