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
- It is written for reproducibility/inspection and may require large time limits.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, List, Tuple

from main import load_instance_data

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


def build_and_solve(
    instance_name: str,
    instances_root: str,
    p: int,
    time_limit: float,
    mip_gap: float,
    threads: int,
    max_scenarios: int = 0,
    max_demands: int = 0,
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
    for scen in scenarios:
        scen.demand_nodes = list(demand_nodes)
        scen.drone_reachability = {
            k: [i for i in v if i in demand_set]
            for k, v in scen.drone_reachability.items()
        }

    if p <= 0 or p > len(depots):
        raise ValueError(f"p must be in [1, {len(depots)}], got {p}")

    # Big-M values.
    max_time = _max_finite_time(scenarios, demand_nodes, depots)
    M_time = (len(demand_nodes) + 2) * max_time * 10.0
    M_order = float(len(demand_nodes))

    model = gp.Model(f"exact_small_{instance_name}")
    model.Params.OutputFlag = 1
    model.Params.TimeLimit = float(time_limit)
    model.Params.MIPGap = float(mip_gap)
    if threads > 0:
        model.Params.Threads = int(threads)

    # -----------------------------
    # First-stage variables
    # -----------------------------
    x = model.addVars(depots, vtype=GRB.BINARY, name="x")
    model.addConstr(gp.quicksum(x[k] for k in depots) == p, name="open_p_depots")

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
            from_nodes = [k] + demand_nodes
            to_nodes = demand_nodes + [sink[k]]
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
    for s in range(num_s):
        for k in depots:
            for idx_i, i in enumerate(demand_nodes):
                for j in demand_nodes[idx_i + 1 :]:
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
            # Depot-open linking.
            for i in demand_nodes:
                model.addConstr(u[s, k, i] <= x[k], name=f"u_open_s{s}_k{k}_i{i}")
                model.addConstr(z[s, k, i] <= x[k], name=f"z_open_s{s}_k{k}_i{i}")
                if not _drone_reachable(scen, k, i):
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
                    y[s, k, p, i]
                    for p in [k] + demand_nodes
                    if (s, k, p, i) in y
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
                for p in demand_nodes:
                    if (s, k, p, i) in y:
                        tpi = scen.truck_times[p][i]
                        model.addConstr(
                            a[s, i] >= a[s, p] + tpi - M_time * (1 - y[s, k, p, i]),
                            name=f"arr_truck_s{s}_k{k}_p{p}_i{i}",
                        )

            # Drone start-time variable active only when assigned.
            for i in demand_nodes:
                model.addConstr(
                    d_start[s, k, i] <= M_time * z[s, k, i],
                    name=f"d_active_s{s}_k{k}_i{i}",
                )
                rt = scen.drone_times[k][i]
                ow = rt / 2.0
                model.addConstr(
                    a[s, i] >= d_start[s, k, i] + ow - M_time * (1 - z[s, k, i]),
                    name=f"arr_drone_s{s}_k{k}_i{i}",
                )

            # Pairwise disjunctive sequencing for single drone at depot k.
            # If both i and j are assigned to this drone, one must be before the other.
            for idx_i, i in enumerate(demand_nodes):
                for j in demand_nodes[idx_i + 1 :]:
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

    model.optimize()

    # -----------------------------
    # Extract result
    # -----------------------------
    result = {
        "instance": instance_name,
        "num_demands": len(demand_nodes),
        "num_scenarios": num_s,
        "depot_base_map": depot_base_map,
        "status": int(model.Status),
        "status_name": model.Status,
        "objective": None,
        "best_bound": None,
        "mip_gap": None,
        "runtime_sec": float(model.Runtime),
        "open_depots": [],
        "open_depots_base": [],
    }

    if model.SolCount > 0:
        result["objective"] = float(model.ObjVal)
        result["best_bound"] = float(model.ObjBound)
        if abs(model.ObjVal) > 1e-9:
            result["mip_gap"] = abs(model.ObjVal - model.ObjBound) / abs(model.ObjVal)
        else:
            result["mip_gap"] = 0.0
        result["open_depots"] = [k for k in depots if x[k].X > 0.5]
        result["open_depots_base"] = [int(depot_base_map.get(k, k)) for k in result["open_depots"]]

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exact Gurobi MILP for small-sized stochastic truck-drone instance."
    )
    parser.add_argument("--instances-root", default="./Instance")
    parser.add_argument("--instance", default="Instance3")
    parser.add_argument("--p", type=int, default=2, help="Number of depots to open.")
    parser.add_argument("--time-limit", type=float, default=3600.0)
    parser.add_argument("--mip-gap", type=float, default=0.0)
    parser.add_argument("--threads", type=int, default=0, help="0 means Gurobi default.")
    parser.add_argument("--max-scenarios", type=int, default=0, help="0 means all scenarios.")
    parser.add_argument("--max-demands", type=int, default=0, help="0 means all demand nodes.")
    parser.add_argument("--out", default="outputs/gurobi_exact_small_result.json")
    args = parser.parse_args()

    result = build_and_solve(
        instance_name=args.instance,
        instances_root=args.instances_root,
        p=args.p,
        time_limit=args.time_limit,
        mip_gap=args.mip_gap,
        threads=args.threads,
        max_scenarios=args.max_scenarios,
        max_demands=args.max_demands,
    )

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Saved: {args.out}")
    print(result)


if __name__ == "__main__":
    main()
