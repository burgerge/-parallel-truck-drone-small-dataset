import heapq
import math
from typing import List, Optional, Tuple


def _truck_time(instance, u: int, v: int) -> float:
    return instance.truck_times.get(u, {}).get(v, math.inf)


def is_truck_edge_feasible(instance, u: int, v: int) -> bool:
    return not math.isinf(_truck_time(instance, u, v))


def remove_first_occurrence(tour: List[int], node: int) -> List[int]:
    if not tour:
        return []
    out = list(tour)
    for idx, val in enumerate(out):
        if val != node:
            continue
        if idx == 0 and val == out[0]:
            continue
        out.pop(idx)
        break
    return out


def best_insertion_position(instance, tour: List[int], node: int) -> Optional[Tuple[int, float]]:
    if not tour:
        return None
    best: Optional[Tuple[int, float]] = None
    for pos in range(1, len(tour) + 1):
        prev = tour[pos - 1]
        nxt = tour[pos] if pos < len(tour) else None
        t_prev_node = _truck_time(instance, prev, node)
        if math.isinf(t_prev_node):
            continue
        if nxt is None:
            delta = t_prev_node
        else:
            t_node_next = _truck_time(instance, node, nxt)
            if math.isinf(t_node_next):
                continue
            t_prev_next = _truck_time(instance, prev, nxt)
            base = t_prev_next if not math.isinf(t_prev_next) else 0.0
            delta = t_prev_node + t_node_next - base

        if best is None or delta < best[1]:
            best = (pos, delta)
    return best


def insert_at(tour: List[int], pos: int, node: int) -> List[int]:
    out = list(tour)
    pos = max(0, min(pos, len(out)))
    out.insert(pos, node)
    return out


def breaks_connectivity(instance, tour: List[int]) -> bool:
    if len(tour) <= 1:
        return False
    for idx in range(1, len(tour)):
        if not is_truck_edge_feasible(instance, tour[idx - 1], tour[idx]):
            return True
    return False


def shortest_path(instance, src: int, dst: int) -> Optional[List[int]]:
    if src == dst:
        return [src]

    pq: List[Tuple[float, int]] = [(0.0, src)]
    dist = {src: 0.0}
    prev = {src: None}

    while pq:
        d, u = heapq.heappop(pq)
        if d > dist.get(u, math.inf):
            continue
        if u == dst:
            break
        for v in instance.adj.get(u, set()):
            w = _truck_time(instance, u, v)
            if math.isinf(w):
                continue
            nd = d + w
            if nd < dist.get(v, math.inf):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))

    if dst not in prev:
        return None

    path = []
    cur = dst
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path
