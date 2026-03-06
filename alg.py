
# alg.py
# ============================================================
# EJOR 2025 (Tureci-Isik et al.) – Parallel Truck–Drone Heuristic
# This file merges the roles of the earlier solver.py + construction.py
# into a single "alg" module that exposes Algorithm 1–5 entry points.
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Iterable, Optional
import copy
import random
import math


# ----------------------------
# Config / helpers
# ----------------------------

@dataclass
class HeuristicConfig:
    """启发式算法的全局参数配置。"""
    # Algorithm 1
    num_depots_to_open: int = 2
    # scenarios used in Algorithm 2 evaluation
    num_scenarios: int = 10
    seed: Optional[int] = None

    # Algorithm 5 (VNS) controls (kept for interface compatibility)
    k_max: int = 2
    l_max: int = 2
    i_max: int = 10

    # Arrival-time computation
    # If your v.txt stores *round-trip* time (common in the paper's case study),
    # set this to True so one-way time is v/2.
    drone_time_is_roundtrip: bool = True
    
    # Objective normalization (paper typically reports average arrival time)
    normalize_by_num_demands: bool = True

    # Infeasibility penalty (used for disconnected truck arcs / uncovered demand, etc.)
    infeasible_penalty: float = 1e9


def _drone_rt(instance, k: int, i: int) -> float:
    """Depot k -> demand i 的往返飞行时间（统一索引方向）。"""
    return instance.drone_times[k][i]


def _one_way_drone_time(instance, k: int, i: int, cfg: HeuristicConfig) -> float:
    """无人机单程时间：根据配置决定是否将往返时间减半。"""
    t = _drone_rt(instance, k, i)
    return (t / 2.0) if cfg.drone_time_is_roundtrip else t


def calculate_arrival_times(truck_tours: Dict[int, List[int]],
                            drone_schedules: Dict[int, List[int]],
                            instance,
                            cfg: Optional[HeuristicConfig] = None) -> float:
    """
    计算单一场景的“延迟类目标值”：所有需求点到达时间之和。

    表达假设（与当前代码一致）：
    - truck_tours[k] 是节点序列 [k, ..., ...]，可能包含重复访问；
      到达时间取第一次到达。
    - drone_schedules[k] 为无人机顺序服务列表；
      每次任务从仓库 k 出发到 i 再返回 k（往返）。
      到达时间 = 无人机当前时钟 + 单程飞行时间。
    """
    if cfg is None:
        cfg = HeuristicConfig()

    demand_nodes = list(instance.demand_nodes)
    reachability = getattr(instance, "drone_reachability", {})
    reachability_sets = {k: set(v) for k, v in reachability.items()}

    # First-arrival times (min across truck/drone if both occur)
    arrival: Dict[int, float] = {i: math.inf for i in demand_nodes}

    # --- Truck arrivals
    for k, tour in truck_tours.items():
        if not tour:
            continue

        # 强制卡车路径从 depot k 开始；若不满足则修复（在前端插入 k）
        if tour[0] != k:
            tour = [k] + list(tour)

        time = 0.0
        prev = k  # 明确起点为 depot k
        arrival.setdefault(k, math.inf)
        arrival[k] = min(arrival.get(k, math.inf), 0.0)

        for node in tour[1:]:
            tt = instance.truck_times.get(prev, {}).get(node, math.inf)
            if math.isinf(tt):
                # 发现断边：本场景不可行，返回严重惩罚值
                return cfg.infeasible_penalty
            time += tt
            if node in arrival:
                arrival[node] = min(arrival[node], time)
            else:
                arrival[node] = time
            prev = node

    # --- Drone arrivals
    for k, seq in drone_schedules.items():
        drone_clock = 0.0
        feasible_set = reachability_sets.get(k)
        for i in seq:
            # one-way time for arrival; round trip for clock increment
            rt = _drone_rt(instance, k, i)
            if feasible_set is not None and i not in feasible_set:
                # Paper Algorithm 5 allows infeasible drone assignments in neighbourhoods;
                # evaluate them with a large dummy duration to keep them undesirable.
                rt = max(rt, cfg.infeasible_penalty)
            ow = (rt / 2.0) if cfg.drone_time_is_roundtrip else rt
            arrival[i] = min(arrival.get(i, math.inf), drone_clock + ow)
            drone_clock += rt

    # If some demand nodes are never served, treat as infeasible (paper-consistent coverage requirement)
    total = 0.0
    for i in demand_nodes:
        if math.isinf(arrival.get(i, math.inf)):
            return cfg.infeasible_penalty
        total += arrival[i]

    if cfg.normalize_by_num_demands and len(demand_nodes) > 0:
        return total / len(demand_nodes)
    return total


# ============================================================
# Drone feasibility sets (Section 3 definitions)
# ============================================================

def _compute_drone_feasible_sets(instance) -> Tuple[Dict[int, Set[int]], Dict[int, List[int]]]:
    """
    返回 (Wi, Dk)：
    - Wi[i]  : 可以由无人机往返、满足续航的仓库集合（drone-eligible depots）。
    - Dk[k]  : depot k 可由无人机服务的需求点，按往返飞行时间非递减排序。

    说明：
    - 优先使用实例文件 D_k.txt 中的 reachability；若缺失，则退化为“往返时间有限”判定。
    - 排序指标使用往返飞行时间 _drone_rt，与论文中 geodesic-based 飞行时间一致。
    """
    Wi: Dict[int, Set[int]] = {i: set() for i in instance.demand_nodes}
    Dk: Dict[int, List[int]] = {}

    depot_nodes = list(getattr(instance, "candidate_depots", [])) or list(instance.drone_times.keys())
    for k in depot_nodes:
        reachable = list(instance.drone_reachability.get(k, []))
        if not reachable:
            # Fallback：视所有往返时间有限的节点为可飞
            reachable = [i for i in instance.demand_nodes if not math.isinf(_drone_rt(instance, k, i))]

        reachable_sorted = sorted(reachable, key=lambda i: _drone_rt(instance, k, i))
        Dk[k] = reachable_sorted
        for i in reachable:
            Wi.setdefault(i, set()).add(k)

    return Wi, Dk


# ============================================================
# Algorithm 4: Backtracking (subroutine of Algorithm 3)
# ============================================================

def algorithm_4_find_backtrack(full_tour: List[int],
                               backtrack_node: int,
                               current_node: int,
                               adj_list: Dict[int, Set[int]]) -> List[int]:
    """
    目的：从 current_node 回溯到 backtrack_node，并尽量走捷径。
    返回值是需要“追加到卡车路径 TTk 末尾”的回溯段。
    """
    # Step 1: Find partial tour segment between backtrack_node and current_node
    # Build segment as it appears in full tour (from backtrack_node ... current_node)
    segment: List[int] = []
    found = False
    # locate last occurrence of current_node (usually at end)
    # and last occurrence of backtrack_node before that
    try:
        cur_idx = len(full_tour) - 1 - full_tour[::-1].index(current_node)
    except ValueError:
        cur_idx = len(full_tour) - 1
    try:
        back_idx = len(full_tour[:cur_idx+1]) - 1 - full_tour[:cur_idx+1][::-1].index(backtrack_node)
    except ValueError:
        back_idx = 0

    if back_idx <= cur_idx:
        segment = full_tour[back_idx:cur_idx+1]
        found = True

    if not found or len(segment) <= 1:
        # fallback: direct hop if exists
        if backtrack_node in adj_list.get(current_node, set()):
            return [backtrack_node]
        return []

    # Step 2: remove loops in the segment (keep a simplified path)
    simplified = list(segment)
    has_revisited = True
    while has_revisited:
        has_revisited = False
        counts = {}
        for n in simplified:
            counts[n] = counts.get(n, 0) + 1
        revisited_nodes = [n for n, c in counts.items() if c > 1]
        if revisited_nodes:
            has_revisited = True
            target = revisited_nodes[0]
            first_idx = simplified.index(target)
            last_idx = len(simplified) - 1 - simplified[::-1].index(target)
            simplified = simplified[:first_idx] + simplified[last_idx:]

    # Step 3: find a shortcut from current_node to any node in simplified (except current_node)
    # simplified is [backtrack_node, ..., current_node]
    target_list = simplified[:-1]
    for i, node in enumerate(target_list):
        if node in adj_list.get(current_node, set()):
            # path becomes current -> node -> ... -> backtrack_node
            path_back_to_root = simplified[:i+1][::-1]  # [node, ..., backtrack_node]
            return path_back_to_root

    # Step 4: no shortcut: follow simplified backwards (excluding current node)
    return simplified[:-1][::-1]


# ============================================================
# Algorithm 3: Initial Construction (truck tour + drone schedule)
# ============================================================

def algorithm_3_initial_construction(X_open: Set[int], instance) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    """
    构造单一场景的初始解：卡车线路（TTk）+ 无人机任务（DSk）。
    按论文 Algorithm 3 的伪代码：
    - 将每个需求点分配给最近的开仓仓库
    - 为每个仓库贪心构造卡车路径
    - 若无直接可达的未访问节点，则回溯
    - 若回溯失败，则将剩余节点加入该仓库的无人机列表
    - 最后先按 Dk 构造可行无人机调度，再追加不可行无人机指派
    """
    _, Dk = _compute_drone_feasible_sets(instance)

    Uk: Dict[int, List[int]] = {k: [] for k in X_open}
    TTk: Dict[int, List[int]] = {k: [] for k in X_open}
    DLk: Dict[int, List[int]] = {k: [] for k in X_open}
    DSk: Dict[int, List[int]] = {k: [] for k in X_open}

    # 2) Assignment: each demand node -> closest open depot.
    # We use drone round-trip time as the available proxy of direct geodesic distance.
    for i in instance.demand_nodes:
        best_k = min(X_open, key=lambda k: _drone_rt(instance, k, i))
        Uk[best_k].append(i)

    # 3) Truck routing per depot
    for k in X_open:
        current = k
        TTk[k].append(current)

        while len(Uk[k]) > 0:
            neighbors = instance.adj.get(current, set())
            feasible_neighbors = [j for j in Uk[k] if j in neighbors]

            if feasible_neighbors:
                # greedy nearest neighbor by truck time
                best_node = min(feasible_neighbors, key=lambda j: instance.truck_times[current][j])
                Uk[k].remove(best_node)
                TTk[k].append(best_node)
                current = best_node
            else:
                # backtracking search
                T = list(TTk[k])
                if T:
                    T.pop()  # remove current node

                stop_flag = 0
                while len(T) > 0:
                    back_node = T[-1]
                    back_neighbors = instance.adj.get(back_node, set())
                    valid_next = [j for j in Uk[k] if j in back_neighbors]
                    if valid_next:
                        Bk = algorithm_4_find_backtrack(TTk[k], back_node, current, instance.adj)
                        TTk[k].extend(Bk)
                        next_node = min(valid_next, key=lambda j: instance.truck_times[back_node][j])
                        Uk[k].remove(next_node)
                        TTk[k].append(next_node)
                        current = next_node
                        stop_flag = 1
                        break
                    T.pop()

                if stop_flag == 0:
                    # Dead-end: assign all remaining nodes to the drone list of the same depot.
                    remaining = list(Uk[k])
                    DLk[k].extend(remaining)
                    Uk[k] = []

    # 4) Drone scheduling:
    #    first append drone-feasible nodes in Dk order,
    #    then append infeasible drone assignments to keep the paper's search state.
    for k in X_open:
        feasible_in_order = [i for i in Dk.get(k, []) if i in DLk[k]]
        infeasible_rest = [i for i in DLk[k] if i not in set(Dk.get(k, []))]
        DSk[k].extend(feasible_in_order)
        DSk[k].extend(infeasible_rest)

    return TTk, DSk


# ============================================================
# Algorithm 5: VNS Improvement
# ============================================================

def algorithm_5_vns_improvement(tt_k: Dict[int, List[int]],
                               ds_k: Dict[int, List[int]],
                               instance,
                               cfg: HeuristicConfig) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    """Algorithm 5：封装 VNS 改进步骤，便于与 Algorithm 2 对接。"""
    from vns.types import Solution
    from vns.vns_engine import improve

    initial = Solution(
        tt={k: list(v) for k, v in tt_k.items()},
        ds={k: list(v) for k, v in ds_k.items()},
    )
    improved = improve(initial, instance, cfg)
    return improved.tt, improved.ds


# ============================================================
# Algorithm 2: Depot Evaluation (average over scenarios)
# ============================================================

def algorithm_2_depot_evaluation(X_open: Set[int],
                                base_instance,
                                scenarios: List,
                                cfg: Optional[HeuristicConfig] = None) -> float:
    """
    Algorithm 2：评估给定开仓集合在多个场景下的期望目标值。
    - 如果 scenarios = [scen1, scen2, ...]：默认等权平均
    - 如果 scenarios = [(scen1, p1), (scen2, p2), ...]：按概率加权
    """
    if cfg is None:
        cfg = HeuristicConfig()

    # detect weighted scenarios
    weighted = False
    if scenarios and isinstance(scenarios[0], tuple) and len(scenarios[0]) == 2:
        weighted = True

    total = 0.0
    weight_sum = 0.0
    for item in scenarios:
        if weighted:
            scen, w = item
            w = float(w)
        else:
            scen, w = item, 1.0

        # Algorithm 2 in the paper evaluates each scenario by:
        # InitialConstruction -> Improvement -> h(R(X, xi))
        tt_k, ds_k = algorithm_3_initial_construction(X_open, scen)
        tt_k2, ds_k2 = algorithm_5_vns_improvement(tt_k, ds_k, scen, cfg)
        total += w * calculate_arrival_times(tt_k2, ds_k2, scen, cfg)
        weight_sum += w

    return total / weight_sum if weight_sum > 0 else cfg.infeasible_penalty


# ============================================================
# Scenario generation helper (used by Algorithm 1 if needed)
# ============================================================

def generate_scenarios_by_time_inflation(base_instance,
                                        num_scenarios: int,
                                        seed: Optional[int] = None,
                                        low: float = 1.0,
                                        high: float = 1.5) -> List:
    """
    场景生成：当没有显式场景文件时，通过膨胀卡车时间模拟随机场景。
    做法：复制基础实例，将每条有限卡车时间乘以 U(low, high)。
    """
    rng = random.Random(seed)
    scenarios = []
    for _ in range(num_scenarios):
        scen = copy.deepcopy(base_instance)
        for i in scen.truck_times:
            for j in scen.truck_times[i]:
                t = scen.truck_times[i][j]
                if t != float('inf') and t > 0:
                    scen.truck_times[i][j] = t * rng.uniform(low, high)
        scenarios.append(scen)
    return scenarios


# ============================================================
# Algorithm 1: Stochastic Location-Routing (swap local search)
# ============================================================

def algorithm_1_stochastic_lrp(instance,
                              cfg: Optional[HeuristicConfig] = None,
                              scenarios: Optional[List] = None) -> Tuple[Set[int], float]:
    """
    Algorithm 1：仓库选址的交换邻域搜索。
    与论文伪代码一致：
    - 随机初始化开仓集合 X
    - 评估整个 swap 邻域 Ndepot(X)
    - 若存在更优邻居，则移动到该轮最优邻居（best-improvement）
    - 否则停止
    返回 (最佳开仓集合, 最佳期望目标值)。
    """
    if cfg is None:
        cfg = HeuristicConfig()
    rng = random.Random(cfg.seed)

    candidates = list(instance.candidate_depots)
    if len(candidates) < cfg.num_depots_to_open:
        X = set(candidates)
    else:
        X = set(rng.sample(candidates, cfg.num_depots_to_open))

    if scenarios is None:
        scenarios = generate_scenarios_by_time_inflation(instance, cfg.num_scenarios, seed=cfg.seed)

    f_min = algorithm_2_depot_evaluation(X, instance, scenarios, cfg)
    best_x = set(X)
    tol = 1e-12

    while True:
        f_new = f_min
        x_prime = set(X)

        # Explore the full swap neighbourhood Ndepot(X) and keep the best improving move.
        for out_node in list(X):
            for in_node in candidates:
                if in_node in X:
                    continue
                Y = set(X)
                Y.remove(out_node)
                Y.add(in_node)

                f_Y = algorithm_2_depot_evaluation(Y, instance, scenarios, cfg)
                if f_Y + tol < f_new:
                    x_prime = Y
                    f_new = f_Y

        if f_new + tol >= f_min:
            break

        X = x_prime
        best_x = set(x_prime)
        f_min = f_new

    return best_x, f_min
