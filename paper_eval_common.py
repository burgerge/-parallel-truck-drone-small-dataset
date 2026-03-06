import json
import math
import os
import statistics
import time
from contextlib import contextmanager
from copy import deepcopy
from typing import Dict, List, Sequence

import alg
from main import load_instance_data


def parse_instance_names(raw: str) -> List[str]:
    raw = (raw or "").strip()
    if not raw:
        return [f"Instance{i}" for i in range(1, 11)]

    if "-" in raw and "," not in raw:
        a, b = raw.split("-", 1)
        start = int(a.strip())
        end = int(b.strip())
        if start > end:
            start, end = end, start
        return [f"Instance{i}" for i in range(start, end + 1)]

    out: List[str] = []
    for token in raw.split(","):
        t = token.strip()
        if not t:
            continue
        if t.lower().startswith("instance"):
            out.append(t)
        else:
            out.append(f"Instance{int(t)}")
    return out


def build_cfg(
    p: int,
    seed: int,
    k_max: int,
    l_max: int,
    i_max: int,
    num_scenarios: int = 10,
) -> alg.HeuristicConfig:
    return alg.HeuristicConfig(
        num_depots_to_open=p,
        num_scenarios=num_scenarios,
        seed=seed,
        k_max=k_max,
        l_max=l_max,
        i_max=i_max,
        drone_time_is_roundtrip=True,
        normalize_by_num_demands=True,
    )


@contextmanager
def no_vns_improvement():
    original = alg.algorithm_5_vns_improvement

    def _noop(tt_k, ds_k, instance, cfg):
        return tt_k, ds_k

    alg.algorithm_5_vns_improvement = _noop
    try:
        yield
    finally:
        alg.algorithm_5_vns_improvement = original


@contextmanager
def excluded_neighborhood(neighborhood_id: int):
    import vns.neighborhoods as nbh

    original = nbh.NEIGHBORHOODS.get(neighborhood_id)

    def _disabled(sol, instance, cfg, rng):
        return []

    nbh.NEIGHBORHOODS[neighborhood_id] = _disabled
    try:
        yield
    finally:
        if original is None:
            nbh.NEIGHBORHOODS.pop(neighborhood_id, None)
        else:
            nbh.NEIGHBORHOODS[neighborhood_id] = original


def run_full_a1(
    instance_name: str,
    instances_root: str,
    cfg: alg.HeuristicConfig,
    disable_improvement: bool = False,
) -> Dict:
    folder = os.path.join(instances_root, instance_name)
    if not os.path.exists(folder):
        raise FileNotFoundError(f"Instance folder not found: {folder}")

    base_instance, scenarios = load_instance_data(folder)
    if not scenarios:
        raise ValueError(f"No scenarios loaded for {instance_name}")
    depot_base_map = dict(getattr(base_instance, "depot_base_map", {}))

    t0 = time.perf_counter()
    if disable_improvement:
        with no_vns_improvement():
            best_x, best_obj = alg.algorithm_1_stochastic_lrp(base_instance, cfg=cfg, scenarios=scenarios)
    else:
        best_x, best_obj = alg.algorithm_1_stochastic_lrp(base_instance, cfg=cfg, scenarios=scenarios)
    elapsed = time.perf_counter() - t0
    best_x_sorted = sorted(best_x)
    best_x_base = [depot_base_map.get(k, k) for k in best_x_sorted]

    return {
        "instance": instance_name,
        "best_x_open": best_x_sorted,
        "best_x_open_base": best_x_base,
        "expected_obj": float(best_obj),
        "runtime_sec": float(elapsed),
        "depot_base_map": depot_base_map,
    }


def ci95(values: Sequence[float]) -> Dict[str, float]:
    arr = list(values)
    if not arr:
        return {"avg": 0.0, "low": 0.0, "high": 0.0}
    avg = statistics.mean(arr)
    if len(arr) == 1:
        return {"avg": avg, "low": avg, "high": avg}
    std = statistics.pstdev(arr)
    half = 1.96 * std / math.sqrt(len(arr))
    return {"avg": avg, "low": avg - half, "high": avg + half}


def ensure_parent(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def save_json(path: str, obj: Dict) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def clone_cfg(cfg: alg.HeuristicConfig) -> alg.HeuristicConfig:
    return deepcopy(cfg)
