import random

from vns.neighborhoods import NEIGHBORHOODS
from vns.objective import objective
from vns.repair import repair_solution
from vns.types import Solution


def shaking(sol: Solution, k: int, instance, cfg, rng: random.Random) -> Solution:
    neigh = NEIGHBORHOODS.get(k, NEIGHBORHOODS[1])
    candidates = neigh(sol, instance, cfg, rng)
    if not candidates:
        return sol
    return rng.choice(candidates)


def _best_neighbor(sol: Solution, neighborhood_id: int, instance, cfg, rng: random.Random):
    neigh = NEIGHBORHOODS.get(neighborhood_id)
    if neigh is None:
        return None, None

    best = None
    best_obj = None
    for cand in neigh(sol, instance, cfg, rng):
        cand_obj = objective(cand, instance, cfg)
        if best is None or cand_obj + 1e-9 < best_obj:
            best = cand
            best_obj = cand_obj
    return best, best_obj


def local_search(sol: Solution, instance, cfg, rng: random.Random) -> Solution:
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
    rng = random.Random(cfg.seed)
    initial_repaired = repair_solution(initial, instance, cfg)
    best = initial_repaired.solution if initial_repaired.solution is not None else initial
    best_obj = objective(best, instance, cfg)

    k_max = min(len(NEIGHBORHOODS), max(1, int(getattr(cfg, "k_max", 1))))
    i_max = max(0, int(getattr(cfg, "i_max", 0)))
    i = 0

    while i <= i_max:
        k = 1
        while k <= k_max:
            shaken = shaking(best, k, instance, cfg, rng)
            local = local_search(shaken, instance, cfg, rng)
            local_obj = objective(local, instance, cfg)
            if local_obj + 1e-9 < best_obj:
                best = local
                best_obj = local_obj
                k = 1
            else:
                k += 1
        i += 1

    return best
