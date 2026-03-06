from typing import List

from vns.drone_ops import can_drone, drone_rt, sort_drone_seq
from vns.truck_ops import (
    best_insertion_position,
    insert_at,
    is_truck_edge_feasible,
    shortest_path,
)
from vns.types import RepairResult, Solution
from vns.validate import validate_solution


def _copy_solution(sol: Solution) -> Solution:
    return Solution(
        tt={k: list(v) for k, v in sol.tt.items()},
        ds={k: list(v) for k, v in sol.ds.items()},
    )


def _ordered_unique(values: List[int]) -> List[int]:
    out = []
    seen = set()
    for v in values:
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def _active_depots(sol: Solution, instance) -> List[int]:
    keys = list(sol.tt.keys()) + list(sol.ds.keys())
    if not keys:
        keys = list(instance.candidate_depots)
    if not keys:
        keys = [0]
    return _ordered_unique(keys)


def repair_solution(sol: Solution, instance, cfg) -> RepairResult:
    work = _copy_solution(sol)
    demand_set = set(instance.demand_nodes)
    notes: List[str] = []
    changed = False
    unassigned: List[int] = []

    depots = _active_depots(work, instance)
    for k in depots:
        if k not in work.tt:
            work.tt[k] = [k]
            changed = True
        if k not in work.ds:
            work.ds[k] = []
            changed = True
        if not work.tt[k]:
            work.tt[k] = [k]
            changed = True
        if work.tt[k][0] != k:
            work.tt[k] = [k] + list(work.tt[k])
            changed = True

    # A) clear infeasible drone demands
    for k in depots:
        cleaned = []
        for d in work.ds.get(k, []):
            if d not in demand_set:
                changed = True
                continue
            if not can_drone(instance, k, d):
                unassigned.append(d)
                changed = True
                continue
            cleaned.append(d)
        work.ds[k] = cleaned

    # B) deduplicate demand services (truck repeated visits are treated as pass-through)
    owner = {}
    for k in depots:
        tour = work.tt[k]
        if not tour:
            work.tt[k] = [k]
            changed = True
            continue
        rebuilt = [tour[0]]
        first_seen_in_tour = set()
        if tour[0] in demand_set:
            owner.setdefault(tour[0], k)
            first_seen_in_tour.add(tour[0])

        for n in tour[1:]:
            if n not in demand_set:
                rebuilt.append(n)
                continue
            if n in first_seen_in_tour:
                # repeated in the same route: pass-through, keep it
                rebuilt.append(n)
                continue
            first_seen_in_tour.add(n)
            if n in owner and owner[n] != k:
                # competing first service in another depot: remove this service occurrence
                changed = True
                continue
            owner.setdefault(n, k)
            rebuilt.append(n)
        work.tt[k] = rebuilt

    seen_demands = set(owner.keys())
    for k in depots:
        rebuilt = []
        seen_in_drone = set()
        for d in work.ds[k]:
            if d not in demand_set:
                changed = True
                continue
            if not can_drone(instance, k, d):
                unassigned.append(d)
                changed = True
                continue
            if d in seen_in_drone:
                changed = True
                continue
            if d in seen_demands:
                changed = True
                continue
            seen_in_drone.add(d)
            seen_demands.add(d)
            rebuilt.append(d)
        work.ds[k] = rebuilt

    # C) fill missing demands by truck insertion first, then drone fallback
    missing = [d for d in instance.demand_nodes if d not in seen_demands]
    to_assign = _ordered_unique(unassigned + missing)
    for d in to_assign:
        if d in seen_demands:
            continue
        best = None  # (delta, depot, pos)
        for k in depots:
            tour = work.tt[k]
            ins = best_insertion_position(instance, tour, d)
            if ins is None:
                continue
            pos, delta = ins
            if best is None or delta < best[0]:
                best = (delta, k, pos)

        if best is not None:
            _, best_k, best_pos = best
            work.tt[best_k] = insert_at(work.tt[best_k], best_pos, d)
            seen_demands.add(d)
            changed = True
            continue

        feasible_drone_depots = [k for k in depots if can_drone(instance, k, d)]
        if feasible_drone_depots:
            best_k = min(feasible_drone_depots, key=lambda k: (drone_rt(instance, k, d), k))
            work.ds[best_k].append(d)
            work.ds[best_k] = sort_drone_seq(instance, best_k, work.ds[best_k])
            seen_demands.add(d)
            changed = True
            continue

        notes.append(f"Cannot assign demand {d} to truck or drone.")
        return RepairResult(solution=None, changed=changed, notes=notes)

    # D) repair truck discontinuities by shortest-path splicing
    for k in depots:
        tour = work.tt[k]
        if len(tour) <= 1:
            continue
        rebuilt = [tour[0]]
        for node in tour[1:]:
            prev = rebuilt[-1]
            if is_truck_edge_feasible(instance, prev, node):
                rebuilt.append(node)
                continue

            path = shortest_path(instance, prev, node)
            if not path or len(path) < 2:
                notes.append(f"Truck edge repair failed for depot {k}: {prev}->{node}")
                return RepairResult(solution=None, changed=changed, notes=notes)
            rebuilt.extend(path[1:])
            changed = True
        work.tt[k] = rebuilt

    repaired = Solution(tt=work.tt, ds=work.ds)
    vr = validate_solution(repaired, instance)
    if not vr.ok:
        notes.extend(vr.errors)
        return RepairResult(solution=None, changed=changed, notes=notes)
    return RepairResult(solution=repaired, changed=changed, notes=notes)
