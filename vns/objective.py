from vns.types import Solution


def objective(sol: Solution, instance, cfg) -> float:
    """
    Unified objective wrapper for VNS modules.
    """
    from alg import calculate_arrival_times

    return calculate_arrival_times(sol.tt, sol.ds, instance, cfg)
