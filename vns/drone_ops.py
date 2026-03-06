import math
from typing import List


def can_drone(instance, depot: int, demand: int) -> bool:
    return demand in set(instance.drone_reachability.get(depot, []))


def drone_rt(instance, depot: int, demand: int) -> float:
    return instance.drone_times.get(depot, {}).get(demand, math.inf)


def sort_drone_seq(instance, depot: int, seq: List[int]) -> List[int]:
    return sorted(seq, key=lambda i: (drone_rt(instance, depot, i), i))
