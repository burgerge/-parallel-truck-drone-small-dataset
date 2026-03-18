import argparse
import heapq
import json
import math
import os
from typing import Dict, List, Optional, Sequence, Set, Tuple

import alg
from gurobi_exact_small_enumeration import _euler_walk_from_arcs, _prepare_data, solve_fixed_combo_commodity


def _parse_open_depots(raw: str) -> List[int]:
    vals = []
    for token in (raw or "").replace(" ", "").split(","):
        if token:
            vals.append(int(token))
    return sorted(vals)


def _shortest_path_allowed(
    scen,
    src: int,
    dst: int,
    allowed_nodes: Set[int],
) -> Optional[List[int]]:
    if src == dst:
        return [src]
    if src not in allowed_nodes or dst not in allowed_nodes:
        return None

    pq: List[Tuple[float, int]] = [(0.0, int(src))]
    dist: Dict[int, float] = {int(src): 0.0}
    prev: Dict[int, Optional[int]] = {int(src): None}

    while pq:
        d, u = heapq.heappop(pq)
        if d > dist.get(u, math.inf):
            continue
        if u == dst:
            break
        for v, w in scen.truck_times.get(u, {}).items():
            v = int(v)
            if v not in allowed_nodes:
                continue
            if math.isinf(float(w)):
                continue
            nd = d + float(w)
            if nd < dist.get(v, math.inf):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))

    if int(dst) not in prev:
        return None

    path: List[int] = []
    cur: Optional[int] = int(dst)
    while cur is not None:
        path.append(int(cur))
        cur = prev.get(cur)
    path.reverse()
    return path


def _normalize_tour(tour: Sequence[int], depot: int) -> List[int]:
    t = [int(x) for x in (tour or [])]
    if not t:
        return [int(depot)]
    if int(t[0]) != int(depot):
        t = [int(depot)] + t
    return t


def _close_tour_by_shortest_path(
    scen,
    tour: Sequence[int],
    depot: int,
    demand_nodes: Sequence[int],
    allowed_nodes: Optional[Set[int]] = None,
) -> List[int]:
    t = _normalize_tour(tour, depot)
    if int(t[-1]) == int(depot):
        return t

    if allowed_nodes is None:
        allowed = set(int(i) for i in demand_nodes)
        allowed.add(int(depot))
    else:
        allowed = set(int(i) for i in allowed_nodes)
        allowed.add(int(depot))
    path_back = _shortest_path_allowed(scen, int(t[-1]), int(depot), allowed)
    if path_back and len(path_back) >= 2:
        return t + [int(x) for x in path_back[1:]]
    return t


def _compress_consecutive_duplicates(tour: Sequence[int]) -> List[int]:
    out: List[int] = []
    for node in tour or []:
        node = int(node)
        if out and int(out[-1]) == int(node):
            continue
        out.append(int(node))
    return out


def _first_visit_demands(tour: Sequence[int], demand_set: Set[int]) -> List[int]:
    seen: Set[int] = set()
    out: List[int] = []
    for node in tour or []:
        node = int(node)
        if node not in demand_set or node in seen:
            continue
        seen.add(node)
        out.append(int(node))
    return out


def _weighted_average(values: Sequence[float], weights: Optional[Sequence[float]]) -> float:
    vals = [float(v) for v in values]
    if not vals:
        return math.nan
    if not weights:
        return sum(vals) / len(vals)
    ws = [float(w) for w in weights]
    s = sum(ws)
    if s <= 0:
        return sum(vals) / len(vals)
    return sum(v * w for v, w in zip(vals, ws)) / s


def _route_degree_diff(arcs: Set[Tuple[int, int]], depot: int) -> Tuple[Dict[int, int], Dict[int, int], Dict[int, int]]:
    indeg: Dict[int, int] = {}
    outdeg: Dict[int, int] = {}
    nodes: Set[int] = {int(depot)}
    for a, b in arcs:
        a = int(a)
        b = int(b)
        nodes.add(a)
        nodes.add(b)
        outdeg[a] = outdeg.get(a, 0) + 1
        indeg[b] = indeg.get(b, 0) + 1
    diff = {int(n): int(outdeg.get(n, 0) - indeg.get(n, 0)) for n in nodes}
    return indeg, outdeg, diff


def _repair_route_arc_balance(
    arcs: Set[Tuple[int, int]],
    depot: int,
    repeated_arc_count: Optional[Dict[Tuple[int, int], int]] = None,
) -> Set[Tuple[int, int]]:
    """
    Make route-arc support compatible with Eq.(12) flow conservation (in-degree == out-degree).

    Table3 truck walks can traverse the same directed arc multiple times, while paper_exact y is binary.
    After collapsing to a set, degree imbalance may appear. We greedily drop arcs from surplus->deficit
    nodes, preferring arcs that were repeated in the original walk.
    """
    cur: Set[Tuple[int, int]] = {(int(a), int(b)) for (a, b) in arcs if int(a) != int(b)}
    rep = repeated_arc_count or {}
    if not cur:
        return cur

    max_iter = max(20, 10 * len(cur))
    for _ in range(max_iter):
        indeg, outdeg, diff = _route_degree_diff(cur, int(depot))
        surplus = [int(n) for n, d in diff.items() if d > 0]
        deficit = [int(n) for n, d in diff.items() if d < 0]
        if not surplus and not deficit:
            return cur

        best_arc: Optional[Tuple[int, int]] = None
        best_score: Optional[Tuple[int, int, int]] = None
        deficit_set = set(deficit)
        for u in surplus:
            for (a, b) in cur:
                if int(a) != int(u):
                    continue
                if int(b) not in deficit_set:
                    continue
                score = 0
                # Prefer removing arcs that were repeated in the original walk.
                if int(rep.get((int(a), int(b)), 0)) > 1:
                    score -= 100
                # Keep depot incident arcs when possible.
                if int(a) == int(depot) or int(b) == int(depot):
                    score += 30
                # Avoid creating isolated tails/heads too aggressively.
                if int(outdeg.get(int(a), 0)) <= 1:
                    score += 20
                if int(indeg.get(int(b), 0)) <= 1:
                    score += 20
                cand = (int(score), int(a), int(b))
                if best_score is None or cand < best_score:
                    best_score = cand
                    best_arc = (int(a), int(b))

        if best_arc is None:
            break
        cur.remove(best_arc)

    # If still unbalanced, clear route arcs and let downstream fallback build a tiny feasible cycle.
    _, _, diff = _route_degree_diff(cur, int(depot))
    if any(int(v) != 0 for v in diff.values()):
        return set()
    return cur


def _truck_arc_ok(scen, a: int, b: int) -> bool:
    return not math.isinf(float(scen.truck_times.get(int(a), {}).get(int(b), math.inf)))


def _choose_promoted_truck_demand(
    scen,
    depot: int,
    preferred_candidates: Sequence[int],
    demand_set: Set[int],
) -> Optional[int]:
    candidates: List[int] = []
    seen: Set[int] = set()
    for seq in (preferred_candidates, sorted(int(i) for i in demand_set)):
        for i in seq:
            i = int(i)
            if i in seen or i not in demand_set:
                continue
            seen.add(i)
            if not (_truck_arc_ok(scen, int(depot), i) and _truck_arc_ok(scen, i, int(depot))):
                continue
            candidates.append(int(i))

    if not candidates:
        return None

    return min(
        candidates,
        key=lambda i: (
            float(scen.truck_times[int(depot)][int(i)]) + float(scen.truck_times[int(i)][int(depot)]),
            int(i),
        ),
    )


def _repair_single_table3_route_for_paper_exact(
    scen,
    depot: int,
    demand_nodes: Sequence[int],
    raw_tour: Sequence[int],
    raw_ds: Sequence[int],
) -> Tuple[List[int], List[int], List[str]]:
    """
    Repair one Table3 truck/drone pair into a route structure that is much closer
    to the paper-exact model semantics:
    - the truck walk is depot-anchored
    - return closure prefers already visited truck nodes, then same-depot drone nodes
    - if a depot would otherwise have no truck movement, promote one local drone
      demand to a tiny truck cycle k->i->k
    - demands touched by the repaired truck walk are removed from the same-depot
      drone sequence to preserve single-service semantics
    """
    depot = int(depot)
    demand_set = set(int(i) for i in demand_nodes)
    notes: List[str] = []

    ds_seq = [int(i) for i in raw_ds if int(i) in demand_set]
    tour = _compress_consecutive_duplicates(int(x) for x in (raw_tour or []))
    if not tour or int(tour[0]) != int(depot):
        tour = [int(depot)] + [int(x) for x in tour]
        tour = _compress_consecutive_duplicates(tour)
        notes.append("prepended_depot")

    if len(tour) <= 1:
        chosen = _choose_promoted_truck_demand(
            scen=scen,
            depot=int(depot),
            preferred_candidates=ds_seq,
            demand_set=demand_set,
        )
        if chosen is not None:
            tour = [int(depot), int(chosen)]
            ds_seq = [int(i) for i in ds_seq if int(i) != int(chosen)]
            notes.append(f"promoted_drone_to_truck:{chosen}")

    # Prefer a closure path that only uses already truck-visited nodes, which
    # avoids pulling new drone-served demands into the truck route.
    allowed_route_only = set(int(x) for x in tour)
    closed = _close_tour_by_shortest_path(
        scen,
        tour,
        int(depot),
        demand_nodes,
        allowed_nodes=allowed_route_only,
    )
    if closed != tour:
        notes.append("closed_on_route_nodes")
    else:
        # If that fails, allow same-depot drone nodes too and reassign any node
        # that becomes truck-visited away from the drone sequence.
        allowed_same_depot = set(int(x) for x in tour) | set(int(i) for i in ds_seq)
        closed2 = _close_tour_by_shortest_path(
            scen,
            tour,
            int(depot),
            demand_nodes,
            allowed_nodes=allowed_same_depot,
        )
        if closed2 != tour:
            closed = closed2
            notes.append("closed_on_route_plus_same_depot_drone_nodes")

    closed = _compress_consecutive_duplicates(closed)
    arc_counts: Dict[Tuple[int, int], int] = {}
    arc_set: Set[Tuple[int, int]] = set()
    for a, b in zip(closed[:-1], closed[1:]):
        a = int(a)
        b = int(b)
        if a == b or not _truck_arc_ok(scen, int(a), int(b)):
            continue
        arc_set.add((int(a), int(b)))
        arc_counts[(int(a), int(b))] = int(arc_counts.get((int(a), int(b)), 0)) + 1

    repaired_arc_set = _repair_route_arc_balance(
        arcs=arc_set,
        depot=int(depot),
        repeated_arc_count=arc_counts,
    )
    if repaired_arc_set and repaired_arc_set != arc_set:
        closed, _unused = _euler_walk_from_arcs(sorted(repaired_arc_set), start=int(depot))
        closed = _compress_consecutive_duplicates(closed)
        notes.append("balanced_collapsed_arc_support")

    truck_demands = _first_visit_demands(closed[1:], demand_set)
    removed = [int(i) for i in ds_seq if int(i) in set(truck_demands)]
    if removed:
        ds_seq = [int(i) for i in ds_seq if int(i) not in set(truck_demands)]
        notes.append("removed_from_drone_due_to_truck_visit:" + ",".join(str(i) for i in removed))

    return [int(x) for x in closed], [int(i) for i in ds_seq], notes


def _repair_table3_paths_for_paper_exact(
    scenarios: Sequence,
    demand_nodes: Sequence[int],
    open_depots: Sequence[int],
    tt_by_s: Sequence[Dict[int, List[int]]],
    ds_by_s: Sequence[Dict[int, List[int]]],
) -> Tuple[List[Dict[int, List[int]]], List[Dict[int, List[int]]], List[Dict[str, object]]]:
    repaired_tt_by_s: List[Dict[int, List[int]]] = []
    repaired_ds_by_s: List[Dict[int, List[int]]] = []
    diagnostics: List[Dict[str, object]] = []
    open_sorted = sorted(int(k) for k in open_depots)

    for s, scen in enumerate(scenarios):
        tt_raw = tt_by_s[s] if s < len(tt_by_s) else {}
        ds_raw = ds_by_s[s] if s < len(ds_by_s) else {}
        tt_fixed: Dict[int, List[int]] = {}
        ds_fixed: Dict[int, List[int]] = {}
        dep_logs: List[Dict[str, object]] = []

        for k in open_sorted:
            truck, drone, notes = _repair_single_table3_route_for_paper_exact(
                scen=scen,
                depot=int(k),
                demand_nodes=demand_nodes,
                raw_tour=tt_raw.get(int(k), [int(k)]),
                raw_ds=ds_raw.get(int(k), []),
            )
            tt_fixed[int(k)] = [int(x) for x in truck]
            ds_fixed[int(k)] = [int(x) for x in drone]
            dep_logs.append(
                {
                    "depot_internal": int(k),
                    "truck_walk_repaired": [int(x) for x in truck],
                    "drone_seq_repaired": [int(x) for x in drone],
                    "repair_notes": list(notes),
                }
            )

        repaired_tt_by_s.append(tt_fixed)
        repaired_ds_by_s.append(ds_fixed)
        diagnostics.append(
            {
                "scenario_index": int(s),
                "depots": dep_logs,
            }
        )

    return repaired_tt_by_s, repaired_ds_by_s, diagnostics


def _extract_table3_paths(t3_payload: Dict) -> Tuple[List[Dict[int, List[int]]], List[Dict[int, List[int]]], List[float]]:
    idx_to_internal_raw = t3_payload.get("depot_idx_to_internal", {})
    idx_to_internal = {int(k): int(v) for k, v in idx_to_internal_raw.items()}

    def _depot_key_to_internal(key: object) -> int:
        kk = int(key)
        return int(idx_to_internal.get(int(kk), int(kk)))

    def _walk_node_to_int(x: object) -> int:
        if isinstance(x, str) and x.startswith("D"):
            d = int(x[1:])
            return int(idx_to_internal.get(int(d), int(d)))
        return int(x)

    # Format A: run_experiment fixed-x output with "scenarios":[{"tt","ds","weight"}, ...]
    if "scenarios" in t3_payload:
        rows = t3_payload.get("scenarios", [])
        tt_by_s: List[Dict[int, List[int]]] = []
        ds_by_s: List[Dict[int, List[int]]] = []
        for r in rows:
            tt = {_depot_key_to_internal(k): [_walk_node_to_int(x) for x in v] for k, v in r.get("tt", {}).items()}
            ds = {_depot_key_to_internal(k): [int(x) for x in v] for k, v in r.get("ds", {}).items()}
            tt_by_s.append(tt)
            ds_by_s.append(ds)
        weights = [float(r.get("weight", 1.0)) for r in rows]
        return tt_by_s, ds_by_s, weights

    # Format B: table3 route dump with "scenario_routes":[{"depots":[...]}]
    if "scenario_routes" in t3_payload:
        rows = t3_payload.get("scenario_routes", [])
        tt_by_s: List[Dict[int, List[int]]] = []
        ds_by_s: List[Dict[int, List[int]]] = []
        for r in rows:
            tt: Dict[int, List[int]] = {}
            ds: Dict[int, List[int]] = {}
            for d in r.get("depots", []):
                k = int(d.get("depot_internal"))
                tt[k] = [int(x) for x in d.get("truck_walk", [])]
                ds[k] = [int(x) for x in d.get("drone_sequence", [])]
            tt_by_s.append(tt)
            ds_by_s.append(ds)
        weights = [1.0 for _ in rows]
        return tt_by_s, ds_by_s, weights

    raise ValueError("Unsupported Table3 path JSON format: expected 'scenarios' or 'scenario_routes'.")


def _extract_gurobi_paths(g_payload: Dict) -> Tuple[List[Dict[int, List[int]]], List[Dict[int, List[int]]]]:
    def _pick_truck_walk(dep_row: Dict) -> List[int]:
        # Prefer route semantics that preserve model order when available.
        for key in ("truck_walk_eval_preferred", "truck_walk_model_order", "truck_walk", "truck_walk_euler"):
            v = dep_row.get(key, [])
            if isinstance(v, list) and v:
                return [int(x) for x in v]
        return []

    rows = g_payload.get("result", {}).get("solution_details", {}).get("scenario_routes", [])
    tt_by_s: List[Dict[int, List[int]]] = []
    ds_by_s: List[Dict[int, List[int]]] = []
    for r in rows:
        tt: Dict[int, List[int]] = {}
        ds: Dict[int, List[int]] = {}
        for d in r.get("depots", []):
            k = int(d.get("depot_internal"))
            tt[k] = _pick_truck_walk(d)
            ds[k] = [int(x) for x in d.get("drone_sequence_spt", [])]
        tt_by_s.append(tt)
        ds_by_s.append(ds)
    return tt_by_s, ds_by_s


def _build_fixed_from_table3_paths(
    scenarios: Sequence,
    demand_nodes: Sequence[int],
    open_depots: Sequence[int],
    tt_by_s: Sequence[Dict[int, List[int]]],
    ds_by_s: Sequence[Dict[int, List[int]]],
) -> Dict[str, Set[Tuple[int, ...]]]:
    demand_set = set(int(i) for i in demand_nodes)
    open_sorted = sorted(int(k) for k in open_depots)

    y_on: Set[Tuple[int, int, int, int]] = set()
    z_on: Set[Tuple[int, int, int]] = set()
    c_on: Set[Tuple[int, int, int]] = set()

    for s, scen in enumerate(scenarios):
        tt_raw = tt_by_s[s] if s < len(tt_by_s) else {}
        ds_raw = ds_by_s[s] if s < len(ds_by_s) else {}
        reach = {int(k): set(int(i) for i in scen.drone_reachability.get(k, [])) for k in open_sorted}

        def _truck_arc_ok(a: int, b: int) -> bool:
            return not math.isinf(float(scen.truck_times.get(int(a), {}).get(int(b), math.inf)))

        tt_closed: Dict[int, List[int]] = {}
        for k in open_sorted:
            tour = tt_raw.get(k, [k])
            tt_closed[k] = _close_tour_by_shortest_path(scen, tour, k, demand_nodes)

        # Build route arcs from closed walks (only feasible arcs).
        y_local: Dict[int, Set[Tuple[int, int]]] = {int(k): set() for k in open_sorted}
        arc_counts_local: Dict[int, Dict[Tuple[int, int], int]] = {int(k): {} for k in open_sorted}
        for k in open_sorted:
            t = tt_closed[k]
            for a, b in zip(t[:-1], t[1:]):
                a = int(a)
                b = int(b)
                if a == b:
                    continue
                if not _truck_arc_ok(a, b):
                    continue
                y_local[int(k)].add((a, b))
                arc_counts_local[int(k)][(a, b)] = int(arc_counts_local[int(k)].get((a, b), 0)) + 1

        # Binary y cannot encode repeated traversal of the same directed arc.
        # Repair degree imbalance caused by arc-set collapsing.
        for k in open_sorted:
            y_local[int(k)] = _repair_route_arc_balance(
                arcs=y_local[int(k)],
                depot=int(k),
                repeated_arc_count=arc_counts_local.get(int(k), {}),
            )

        # Drone owner: first-come by depot order, but keep only endurance-feasible assignments.
        drone_owner: Dict[int, int] = {}
        for k in open_sorted:
            for i in ds_raw.get(k, []):
                i = int(i)
                if i not in demand_set:
                    continue
                if i in drone_owner:
                    continue
                if i not in reach.get(int(k), set()):
                    continue
                drone_owner[i] = int(k)

        # truck owner: first visit in each route, excluding drone-assigned demands
        truck_owner: Dict[int, int] = {}
        for k in open_sorted:
            seen_local: Set[int] = set()
            for node in tt_closed[k][1:]:
                node = int(node)
                if node not in demand_set or node in seen_local:
                    continue
                seen_local.add(node)
                if node in drone_owner:
                    continue
                if node not in truck_owner:
                    truck_owner[node] = int(k)

        # If a depot has no truck arcs at all, create a tiny feasible cycle k->i->k.
        # This avoids paper-flow infeasibility (Eq.17/Eq.20) when all demand is put on drones.
        for k in open_sorted:
            if y_local[int(k)]:
                continue
            candidates = [int(i) for i in ds_raw.get(k, []) if int(i) in demand_set]
            candidates.extend(sorted(int(i) for i in demand_set if int(i) not in candidates))
            chosen = None
            for i in candidates:
                if _truck_arc_ok(k, i) and _truck_arc_ok(i, k):
                    chosen = int(i)
                    break
            if chosen is not None:
                y_local[int(k)].add((int(k), chosen))
                y_local[int(k)].add((chosen, int(k)))
                truck_owner[chosen] = int(k)
                if chosen in drone_owner:
                    del drone_owner[chosen]

        # Any demand node that appears in truck arcs must be truck-assigned (Eq.7 consistency).
        # If the same demand is used as transit by multiple routes, keep one owner route and
        # drop incident arcs from other routes to preserve single-assignment feasibility.
        incident_ks: Dict[int, Set[int]] = {}
        for k in open_sorted:
            for a, b in y_local[int(k)]:
                if a in demand_set:
                    incident_ks.setdefault(int(a), set()).add(int(k))
                if b in demand_set:
                    incident_ks.setdefault(int(b), set()).add(int(k))

        for i, ks in sorted(incident_ks.items()):
            ks_sorted = sorted(int(k) for k in ks)
            owner = truck_owner.get(int(i))
            if owner not in ks:
                owner = int(ks_sorted[0])
            truck_owner[int(i)] = int(owner)
            if int(i) in drone_owner:
                del drone_owner[int(i)]

            for k in ks_sorted:
                if int(k) == int(owner):
                    continue
                y_local[int(k)] = {(a, b) for (a, b) in y_local[int(k)] if a != int(i) and b != int(i)}

        # Ensure each route has a return arc to its depot.
        for k in open_sorted:
            has_return = any(int(b) == int(k) for (_, b) in y_local[int(k)])
            if has_return:
                continue
            dep_targets = [int(j) for (a, j) in y_local[int(k)] if int(a) == int(k)]
            chosen = dep_targets[0] if dep_targets else None
            if chosen is None:
                for i in sorted(demand_set):
                    if _truck_arc_ok(k, i) and _truck_arc_ok(i, k):
                        chosen = int(i)
                        y_local[int(k)].add((int(k), chosen))
                        break
            if chosen is not None and _truck_arc_ok(chosen, k):
                y_local[int(k)].add((int(chosen), int(k)))
                truck_owner[int(chosen)] = int(k)
                if int(chosen) in drone_owner:
                    del drone_owner[int(chosen)]

        # fallback for uncovered demand (should be rare)
        for i in sorted(demand_set):
            if i in drone_owner or i in truck_owner:
                continue
            chosen: Optional[int] = None
            for k in open_sorted:
                if i in tt_closed[k][1:]:
                    chosen = int(k)
                    break
            if chosen is None:
                # Prefer a feasible drone owner if possible.
                for k in open_sorted:
                    if i in reach.get(int(k), set()):
                        chosen = None
                        drone_owner[int(i)] = int(k)
                        break
            if chosen is not None:
                truck_owner[int(i)] = int(chosen)

        # Final consistency: incident-demand cannot remain drone-assigned.
        incident_final: Dict[int, Set[int]] = {}
        for k in open_sorted:
            for a, b in y_local[int(k)]:
                if a in demand_set:
                    incident_final.setdefault(int(a), set()).add(int(k))
                if b in demand_set:
                    incident_final.setdefault(int(b), set()).add(int(k))
        for i, ks in incident_final.items():
            if i in drone_owner:
                del drone_owner[int(i)]
            if i not in truck_owner:
                truck_owner[int(i)] = int(sorted(ks)[0])

        # Emit scenario-fixed binaries.
        for k in open_sorted:
            for a, b in y_local[int(k)]:
                y_on.add((int(s), int(k), int(a), int(b)))
        for i, k in truck_owner.items():
            c_on.add((int(s), int(k), int(i)))
        for i, k in drone_owner.items():
            if int(i) in truck_owner:
                continue
            z_on.add((int(s), int(k), int(i)))

    return {"y_on": y_on, "z_on": z_on, "c_on": c_on}


def _build_strict_fixed_from_table3_paths(
    scenarios: Sequence,
    demand_nodes: Sequence[int],
    open_depots: Sequence[int],
    tt_by_s: Sequence[Dict[int, List[int]]],
    ds_by_s: Sequence[Dict[int, List[int]]],
) -> Dict[str, Set[Tuple[int, ...]]]:
    """
    Build fixed binaries that preserve explicit truck-walk order as closely as
    the exact model allows.

    Compared with `_build_fixed_from_table3_paths`, this also fixes:
    - `w_on`: first departure from each truck-served demand
    - `a_on`: last departure from each truck-served demand
    - `b_on`: predecessor-successor relation for non-first departures

    The truck walk is normalized before encoding:
    - prepend depot if missing
    - remove consecutive duplicates
    - close back to depot by shortest feasible path if needed
    """
    demand_set = set(int(i) for i in demand_nodes)
    open_sorted = sorted(int(k) for k in open_depots)

    fixed: Dict[str, Set[Tuple[int, ...]]] = {
        "y_on": set(),
        "z_on": set(),
        "c_on": set(),
        "w_on": set(),
        "a_on": set(),
        "b_on": set(),
    }

    repaired_tt_by_s, repaired_ds_by_s, _diagnostics = _repair_table3_paths_for_paper_exact(
        scenarios=scenarios,
        demand_nodes=demand_nodes,
        open_depots=open_depots,
        tt_by_s=tt_by_s,
        ds_by_s=ds_by_s,
    )

    for s, scen in enumerate(scenarios):
        tt_raw = repaired_tt_by_s[s] if s < len(repaired_tt_by_s) else {}
        ds_raw = repaired_ds_by_s[s] if s < len(repaired_ds_by_s) else {}

        for k in open_sorted:
            closed_tour = [int(x) for x in tt_raw.get(int(k), [int(k)])]
            closed_tour = _compress_consecutive_duplicates(closed_tour)
            route_nodes = demand_set | {int(k)}

            for a, b in zip(closed_tour[:-1], closed_tour[1:]):
                a = int(a)
                b = int(b)
                if a == b:
                    continue
                if a not in route_nodes or b not in route_nodes:
                    continue
                if math.isinf(float(scen.truck_times.get(int(a), {}).get(int(b), math.inf))):
                    continue
                fixed["y_on"].add((int(s), int(k), int(a), int(b)))

            truck_demands = _first_visit_demands(closed_tour[1:], demand_set)
            truck_demand_set = set(int(i) for i in truck_demands)

            for i in ds_raw.get(int(k), []):
                i = int(i)
                if i in demand_set and i not in truck_demand_set:
                    fixed["z_on"].add((int(s), int(k), int(i)))

            for i in truck_demands:
                fixed["c_on"].add((int(s), int(k), int(i)))

            occurrences: Dict[int, List[Tuple[int, int]]] = {}
            for pos in range(1, len(closed_tour) - 1):
                prev_node = int(closed_tour[pos - 1])
                cur_node = int(closed_tour[pos])
                next_node = int(closed_tour[pos + 1])
                if cur_node not in demand_set:
                    continue
                if (int(s), int(k), int(cur_node), int(next_node)) not in fixed["y_on"]:
                    continue
                occurrences.setdefault(int(cur_node), []).append((int(prev_node), int(next_node)))

            for i, occs in occurrences.items():
                first_pred, first_next = occs[0]
                _ = first_pred
                fixed["w_on"].add((int(s), int(k), int(i), int(first_next)))

                last_pred, last_next = occs[-1]
                _ = last_pred
                fixed["a_on"].add((int(s), int(k), int(i), int(last_next)))

                for pred, nxt in occs[1:]:
                    if (int(s), int(k), int(pred), int(i)) not in fixed["y_on"]:
                        continue
                    if (int(s), int(k), int(i), int(nxt)) not in fixed["y_on"]:
                        continue
                    fixed["b_on"].add((int(s), int(pred), int(i), int(nxt)))

    return fixed


def _build_fixed_from_gurobi_paths(g_payload: Dict) -> Dict[str, Set[Tuple[int, ...]]]:
    y_on: Set[Tuple[int, int, int, int]] = set()
    z_on: Set[Tuple[int, int, int]] = set()
    c_on: Set[Tuple[int, int, int]] = set()

    rows = g_payload.get("result", {}).get("solution_details", {}).get("scenario_routes", [])
    for r in rows:
        s = int(r.get("scenario_index"))
        for d in r.get("depots", []):
            k = int(d.get("depot_internal"))
            for a, b in d.get("truck_active_arcs", []):
                a = int(a)
                b = int(b)
                if a != b:
                    y_on.add((s, k, a, b))
            for i in d.get("drone_assigned_demands", []):
                z_on.add((s, k, int(i)))
            for i in d.get("truck_assigned_demands", []):
                c_on.add((s, k, int(i)))

    return {"y_on": y_on, "z_on": z_on, "c_on": c_on}


def _evaluate_by_arrival_sim(
    scenarios: Sequence,
    tt_by_s: Sequence[Dict[int, List[int]]],
    ds_by_s: Sequence[Dict[int, List[int]]],
    weights: Optional[Sequence[float]],
) -> Dict:
    cfg = alg.HeuristicConfig(
        num_depots_to_open=2,
        num_scenarios=len(scenarios),
        seed=123,
        k_max=7,
        l_max=6,
        i_max=5,
        drone_time_is_roundtrip=True,
        normalize_by_num_demands=True,
        strict_feasibility=True,
    )
    vals: List[float] = []
    for s, scen in enumerate(scenarios):
        tt = tt_by_s[s] if s < len(tt_by_s) else {}
        ds = ds_by_s[s] if s < len(ds_by_s) else {}
        vals.append(float(alg.calculate_arrival_times(tt, ds, scen, cfg)))
    return {
        "scenario_values": vals,
        "weighted_avg": _weighted_average(vals, weights),
        "simple_avg": (sum(vals) / len(vals)) if vals else math.nan,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified scoring comparison: evaluate Table3 and Gurobi paths under the same metrics."
    )
    parser.add_argument("--instance", required=True)
    parser.add_argument("--instances-root", default="./Instance")
    parser.add_argument("--open-depots", required=True, help="Internal depot ids, e.g. 15,17")
    parser.add_argument("--table3-path-json", required=True, help="run_experiment fixed-x output JSON")
    parser.add_argument("--gurobi-path-json", required=True, help="gurobi route JSON with solution_details")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    open_depots = _parse_open_depots(args.open_depots)
    if not open_depots:
        raise ValueError("--open-depots is empty.")

    base, scenarios, demand_nodes, _ = _prepare_data(args.instance, args.instances_root, 0, 0)

    t3_payload = json.load(open(args.table3_path_json, "r", encoding="utf-8"))
    g_payload = json.load(open(args.gurobi_path_json, "r", encoding="utf-8"))

    t3_tt_by_s, t3_ds_by_s, weights = _extract_table3_paths(t3_payload)
    g_tt_by_s, g_ds_by_s = _extract_gurobi_paths(g_payload)

    fixed_t3 = _build_fixed_from_table3_paths(
        scenarios=scenarios,
        demand_nodes=demand_nodes,
        open_depots=open_depots,
        tt_by_s=t3_tt_by_s,
        ds_by_s=t3_ds_by_s,
    )
    fixed_g = _build_fixed_from_gurobi_paths(g_payload)

    t3_on_paper = solve_fixed_combo_commodity(
        instance_name=args.instance,
        scenarios=scenarios,
        demand_nodes=demand_nodes,
        open_depots=open_depots,
        time_limit=1800.0,
        mip_gap=0.0,
        mip_gap_abs=None,
        threads=0,
        output_flag=0,
        mip_focus=-1,
        heuristics=-1.0,
        presolve=-1,
        cuts=-1,
        symmetry=-1,
        return_solution_details=False,
        fixed_binary_values=fixed_t3,
    )
    g_on_paper = solve_fixed_combo_commodity(
        instance_name=args.instance,
        scenarios=scenarios,
        demand_nodes=demand_nodes,
        open_depots=open_depots,
        time_limit=1800.0,
        mip_gap=0.0,
        mip_gap_abs=None,
        threads=0,
        output_flag=0,
        mip_focus=-1,
        heuristics=-1.0,
        presolve=-1,
        cuts=-1,
        symmetry=-1,
        return_solution_details=False,
        fixed_binary_values=fixed_g,
    )

    t3_on_arrival = _evaluate_by_arrival_sim(
        scenarios=scenarios,
        tt_by_s=t3_tt_by_s,
        ds_by_s=t3_ds_by_s,
        weights=weights,
    )
    g_on_arrival = _evaluate_by_arrival_sim(
        scenarios=scenarios,
        tt_by_s=g_tt_by_s,
        ds_by_s=g_ds_by_s,
        weights=weights,
    )

    out_obj = {
        "instance": args.instance,
        "open_depots_internal": open_depots,
        "weights": weights,
        "table3_path": {
            "paper_exact_score": {
                "status": t3_on_paper.get("status_name"),
                "objective": t3_on_paper.get("objective"),
                "best_bound": t3_on_paper.get("best_bound"),
                "iis_count": t3_on_paper.get("iis_count"),
                "iis_top": t3_on_paper.get("iis_top", []),
            },
            "arrival_sim_score": t3_on_arrival,
        },
        "gurobi_path": {
            "paper_exact_score": {
                "status": g_on_paper.get("status_name"),
                "objective": g_on_paper.get("objective"),
                "best_bound": g_on_paper.get("best_bound"),
            },
            "arrival_sim_score": g_on_arrival,
            "original_gurobi_objective_from_file": g_payload.get("result", {}).get("objective"),
        },
        "notes": [
            "paper_exact_score uses Eq.(4)-style objective in solve_fixed_combo_commodity with fixed y/z/c binaries.",
            "arrival_sim_score uses alg.calculate_arrival_times on explicit TT/DS paths.",
            "If table3 path is INFEASIBLE under paper_exact_score, path semantics differ from exact-model-feasible route structure.",
        ],
    }

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, ensure_ascii=False, indent=2)
    print(f"Saved unified scoring report: {args.out}")


if __name__ == "__main__":
    main()
