import random
from typing import Callable, List

from vns.drone_ops import drone_rt
from vns.truck_ops import best_insertion_position, insert_at, remove_first_occurrence
from vns.types import Solution

NeighborhoodFn = Callable[[Solution, object, object, random.Random], List[Solution]]


def _copy_solution(sol: Solution) -> Solution:
    return Solution(
        tt={k: list(v) for k, v in sol.tt.items()},
        ds={k: list(v) for k, v in sol.ds.items()},
    )


def _signature(sol: Solution) -> str:
    tt = tuple((k, tuple(v)) for k, v in sorted(sol.tt.items()))
    ds = tuple((k, tuple(v)) for k, v in sorted(sol.ds.items()))
    return str((tt, ds))


def _collect(raw: Solution, instance, cfg, out: List[Solution], seen: set) -> None:
    sig = _signature(raw)
    if sig in seen:
        return
    seen.add(sig)
    out.append(raw)


def _all_depots(sol: Solution) -> List[int]:
    return list(dict.fromkeys(list(sol.tt.keys()) + list(sol.ds.keys())))


def _closest_drone_depot(instance, depots: List[int], node: int) -> int:
    return min(depots, key=lambda k: drone_rt(instance, k, node))


def n1_truck_to_truck_reassign(sol: Solution, instance, cfg, rng: random.Random) -> List[Solution]:
    out: List[Solution] = []
    seen = set()
    demand_set = set(instance.demand_nodes)
    depots = _all_depots(sol)
    for a in depots:
        tour_a = sol.tt.get(a, [a])
        if len(tour_a) <= 1:
            continue
        for node in [n for n in tour_a[1:] if n in demand_set]:
            for b in depots:
                if b == a:
                    continue
                tour_b = sol.tt.get(b, [b])
                ins = best_insertion_position(instance, tour_b, node)
                if ins is None:
                    continue
                pos, _ = ins
                cand = _copy_solution(sol)
                cand.tt.setdefault(a, [a])
                cand.tt.setdefault(b, [b])
                cand.tt[a] = remove_first_occurrence(cand.tt[a], node)
                cand.tt[b] = insert_at(cand.tt[b], pos, node)
                _collect(cand, instance, cfg, out, seen)
    return out


def n2_truck_to_drone_same_depot(sol: Solution, instance, cfg, rng: random.Random) -> List[Solution]:
    out: List[Solution] = []
    seen = set()
    demand_set = set(instance.demand_nodes)
    for k, tour in sol.tt.items():
        for node in [n for n in tour[1:] if n in demand_set]:
            cand = _copy_solution(sol)
            cand.tt[k] = remove_first_occurrence(cand.tt[k], node)
            cand.ds.setdefault(k, []).append(node)
            _collect(cand, instance, cfg, out, seen)
    return out


def n3_truck_to_drone_cross_depot(sol: Solution, instance, cfg, rng: random.Random) -> List[Solution]:
    out: List[Solution] = []
    seen = set()
    demand_set = set(instance.demand_nodes)
    depots = _all_depots(sol)
    for a, tour in sol.tt.items():
        for node in [n for n in tour[1:] if n in demand_set]:
            for b in depots:
                if a == b:
                    continue
                cand = _copy_solution(sol)
                cand.tt[a] = remove_first_occurrence(cand.tt[a], node)
                cand.ds.setdefault(b, []).append(node)
                _collect(cand, instance, cfg, out, seen)
    return out


def n4_drone_to_truck_same_depot(sol: Solution, instance, cfg, rng: random.Random) -> List[Solution]:
    out: List[Solution] = []
    seen = set()
    for k, seq in sol.ds.items():
        for node in list(seq):
            ins = best_insertion_position(instance, sol.tt.get(k, [k]), node)
            if ins is None:
                continue
            pos, _ = ins
            cand = _copy_solution(sol)
            cand.ds[k].remove(node)
            cand.tt.setdefault(k, [k])
            cand.tt[k] = insert_at(cand.tt[k], pos, node)
            _collect(cand, instance, cfg, out, seen)
    return out


def n5_drone_to_truck_cross_depot(sol: Solution, instance, cfg, rng: random.Random) -> List[Solution]:
    out: List[Solution] = []
    seen = set()
    depots = _all_depots(sol)
    for a, seq in sol.ds.items():
        for node in list(seq):
            for b in depots:
                if a == b:
                    continue
                ins = best_insertion_position(instance, sol.tt.get(b, [b]), node)
                if ins is None:
                    continue
                pos, _ = ins
                cand = _copy_solution(sol)
                cand.ds[a].remove(node)
                cand.tt.setdefault(b, [b])
                cand.tt[b] = insert_at(cand.tt[b], pos, node)
                _collect(cand, instance, cfg, out, seen)
    return out


def n6_drone_to_drone_reassign(sol: Solution, instance, cfg, rng: random.Random) -> List[Solution]:
    out: List[Solution] = []
    seen = set()
    depots = _all_depots(sol)
    for a, seq in sol.ds.items():
        for node in list(seq):
            for b in depots:
                if a == b:
                    continue
                cand = _copy_solution(sol)
                cand.ds[a].remove(node)
                cand.ds.setdefault(b, []).append(node)
                _collect(cand, instance, cfg, out, seen)
    return out


def n7_shaking_reconstruct_truck(sol: Solution, instance, cfg, rng: random.Random) -> List[Solution]:
    out: List[Solution] = []
    seen = set()
    demand_set = set(instance.demand_nodes)
    depots = _all_depots(sol)
    for k in list(sol.tt.keys()):
        old_tour = sol.tt.get(k, [k])
        old_nodes = [n for n in old_tour[1:] if n in demand_set]
        if not old_nodes:
            continue
        for start_node in old_nodes:
            cand = _copy_solution(sol)
            cand.tt[k] = [k]

            remaining = set(old_nodes)
            if start_node in instance.adj.get(k, set()):
                cand.tt[k].append(start_node)
                remaining.remove(start_node)

            while remaining:
                last = cand.tt[k][-1]
                feasible = [n for n in remaining if n in instance.adj.get(last, set())]
                if not feasible:
                    break
                nxt = min(feasible, key=lambda n: instance.truck_times[last].get(n, float("inf")))
                cand.tt[k].append(nxt)
                remaining.remove(nxt)

            stayed_on_truck = set(cand.tt[k][1:])
            leftovers = [n for n in old_nodes if n not in stayed_on_truck]
            for node in leftovers:
                target_depot = _closest_drone_depot(instance, depots, node)
                cand.ds.setdefault(target_depot, []).append(node)

            _collect(cand, instance, cfg, out, seen)
    return out


NEIGHBORHOODS = {
    1: n1_truck_to_truck_reassign,
    2: n2_truck_to_drone_same_depot,
    3: n3_truck_to_drone_cross_depot,
    4: n4_drone_to_truck_same_depot,
    5: n5_drone_to_truck_cross_depot,
    6: n6_drone_to_drone_reassign,
    7: n7_shaking_reconstruct_truck,
}
