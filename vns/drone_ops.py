import math
from typing import List


def can_drone(instance, depot: int, demand: int) -> bool:
    """判断 demand 是否在 depot 的无人机可达集合中。"""
    return demand in set(instance.drone_reachability.get(depot, []))


def drone_rt(instance, depot: int, demand: int) -> float:
    """读取无人机 depot->demand 的往返时间。"""
    return instance.drone_times.get(depot, {}).get(demand, math.inf)


def sort_drone_seq(instance, depot: int, seq: List[int]) -> List[int]:
    """
    按无人机往返时间对任务序列排序（同时间按节点号）。

    该顺序用于构造或修复后，让无人机序列具有稳定、可解释的服务次序。
    """
    return sorted(seq, key=lambda i: (drone_rt(instance, depot, i), i))
