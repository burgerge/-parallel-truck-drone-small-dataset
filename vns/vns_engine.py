import random

from vns.neighborhoods import NEIGHBORHOODS
from vns.objective import objective
from vns.repair import repair_solution
from vns.types import Solution


def shaking(sol: Solution, k: int, instance, cfg, rng: random.Random) -> Solution:
    """
    A5 的 shaking 步骤。

    输入当前解 sol 和邻域编号 k，从 Nk(sol) 中随机抽取一个候选解。
    若 Nk 为空，则直接返回原解。
    """
    neigh = NEIGHBORHOODS.get(k, NEIGHBORHOODS[1])
    candidates = neigh(sol, instance, cfg, rng)
    if not candidates:
        return sol
    return rng.choice(candidates)


def _strict_feasibility_enabled(cfg) -> bool:
    return bool(getattr(cfg, "strict_feasibility", True))


def _repair_or_none(sol: Solution, instance, cfg):
    rr = repair_solution(sol, instance, cfg)
    return rr.solution


def _best_neighbor(sol: Solution, neighborhood_id: int, instance, cfg, rng: random.Random):
    """
    在指定邻域内做 best-improvement 选择。

    返回：
    - best: 邻域中目标值最优的候选解
    - best_obj: 对应目标值
    """
    neigh = NEIGHBORHOODS.get(neighborhood_id)
    if neigh is None:
        return None, None

    best = None
    best_obj = None
    strict = _strict_feasibility_enabled(cfg)
    for cand in neigh(sol, instance, cfg, rng):
        candidate = cand
        if strict:
            candidate = _repair_or_none(cand, instance, cfg)
            if candidate is None:
                continue
        cand_obj = objective(candidate, instance, cfg)
        if best is None or cand_obj + 1e-9 < best_obj:
            best = candidate
            best_obj = cand_obj
    return best, best_obj


def local_search(sol: Solution, instance, cfg, rng: random.Random) -> Solution:
    """
    A5 的局部搜索阶段（固定 shaking 后的深挖）。

    策略：
    - local neighborhood 依次使用 N1..N6（论文中 N7 仅用于 shaking）；
    - 每个邻域内部执行 best-improvement 下降，直到该邻域无法改进；
    - 若当前邻域有改进，重置到 N1；否则切到下一个邻域。
    """
    strict = _strict_feasibility_enabled(cfg)
    if strict:
        repaired = _repair_or_none(sol, instance, cfg)
        current = repaired if repaired is not None else sol
    else:
        current = sol
    current_obj = objective(current, instance, cfg)
    # Paper: N7 is used only in shaking, local search uses N1..N6.
    l_max = min(6, max(1, int(getattr(cfg, "l_max", 1))))
    l = 1

    while l <= l_max:
        improved_in_neighborhood = False

        # Best-improvement descent within the same local-search neighborhood.
        while True:
            best, best_obj = _best_neighbor(current, l, instance, cfg, rng)
            if best is None or best_obj is None or best_obj + 1e-9 >= current_obj:
                break
            current = best
            current_obj = best_obj
            improved_in_neighborhood = True

        if improved_in_neighborhood:
            l = 1
        else:
            l += 1
    return current


def improve(initial: Solution, instance, cfg) -> Solution:
    """
    Algorithm 5 总控流程（VNS）。

    结构：
    1) 先对初始解做 repair（保证尽量可行）；
    2) 外层 i 循环控制最大迭代次数；
    3) 内层 k 循环控制 shaking 邻域；
    4) 每次 shaking 后接 local_search；
    5) 若找到更优解则接受并将 k 重置为 1。

    这是 alg.py::algorithm_5_vns_improvement 调用的核心实现。
    """
    rng = random.Random(cfg.seed)
    strict = _strict_feasibility_enabled(cfg)
    if strict:
        initial_repaired = repair_solution(initial, instance, cfg)
        best = initial_repaired.solution if initial_repaired.solution is not None else initial
        if initial_repaired.solution is None:
            # No feasible seed can be constructed; keep original and let caller penalize if needed.
            return initial
    else:
        # Paper heuristic allows temporarily infeasible route assignments and
        # drives search by penalized objective values; do not force early repair.
        best = initial
    best_obj = objective(best, instance, cfg)

    k_max = min(len(NEIGHBORHOODS), max(1, int(getattr(cfg, "k_max", 1))))
    i_max = max(0, int(getattr(cfg, "i_max", 0)))
    i = 0

    while i <= i_max:
        k = 1
        while k <= k_max:
            shaken = shaking(best, k, instance, cfg, rng)
            if strict:
                shaken_repaired = _repair_or_none(shaken, instance, cfg)
                if shaken_repaired is None:
                    k += 1
                    continue
                shaken = shaken_repaired
            local = local_search(shaken, instance, cfg, rng)
            if strict:
                local_repaired = _repair_or_none(local, instance, cfg)
                if local_repaired is None:
                    k += 1
                    continue
                local = local_repaired
            local_obj = objective(local, instance, cfg)
            if local_obj + 1e-9 < best_obj:
                best = local
                best_obj = local_obj
                k = 1
            else:
                k += 1
        i += 1

    return best
