from vns.types import Solution


def objective(sol: Solution, instance, cfg) -> float:
    """
    VNS 统一目标函数接口。

    说明：
    - vns 子模块内部统一调用 objective(...) 比较候选解优劣；
    - 实际目标定义不在 vns 内重复实现，而是复用 alg.calculate_arrival_times，
      保证 A2/A5 与主流程目标口径完全一致。
    """
    from alg import calculate_arrival_times

    return calculate_arrival_times(sol.tt, sol.ds, instance, cfg)
