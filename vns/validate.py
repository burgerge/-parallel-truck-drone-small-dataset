import math

from vns.types import Solution, ValidationResult


def validate_solution(sol: Solution, instance) -> ValidationResult:
    """
    校验解的可行性与一致性。

    检查项：
    - 每条卡车路径是否以对应 depot 开始；
    - 卡车边是否可行；
    - 无人机任务是否在可达集合内；
    - 每个需求点是否“恰好服务一次”（不重复、不遗漏）。

    返回 ValidationResult，供 repair 与审计脚本复用。
    """
    demand_set = set(instance.demand_nodes)
    counts = {i: 0 for i in demand_set}
    errors = []
    infeasible_truck_edges = []
    infeasible_drone_assign = []

    for depot, tour in sol.tt.items():
        if not tour:
            errors.append(f"Truck tour of depot {depot} is empty.")
            continue
        if tour[0] != depot:
            errors.append(f"Truck tour of depot {depot} does not start at depot.")

        first_seen_in_tour = set()
        for idx, node in enumerate(tour):
            if node in demand_set and node not in first_seen_in_tour:
                counts[node] += 1
                first_seen_in_tour.add(node)
            if idx == 0:
                continue
            u = tour[idx - 1]
            v = node
            t = instance.truck_times.get(u, {}).get(v, math.inf)
            if math.isinf(t):
                infeasible_truck_edges.append((depot, u, v))

    for depot, seq in sol.ds.items():
        reachable = set(instance.drone_reachability.get(depot, []))
        for demand in seq:
            if demand not in demand_set:
                errors.append(f"Drone of depot {depot} serves non-demand node {demand}.")
                continue
            counts[demand] += 1
            if demand not in reachable:
                infeasible_drone_assign.append((depot, demand))

    duplicated = {i for i, c in counts.items() if c > 1}
    missing = {i for i, c in counts.items() if c == 0}

    if infeasible_truck_edges:
        errors.append(f"{len(infeasible_truck_edges)} infeasible truck edges detected.")
    if infeasible_drone_assign:
        errors.append(f"{len(infeasible_drone_assign)} infeasible drone assignments detected.")
    if duplicated:
        errors.append(f"Duplicated demand nodes: {sorted(duplicated)}")
    if missing:
        errors.append(f"Missing demand nodes: {sorted(missing)}")

    ok = not errors and not infeasible_truck_edges and not infeasible_drone_assign and not duplicated and not missing
    return ValidationResult(
        ok=ok,
        errors=errors,
        missing=missing,
        duplicated=duplicated,
        infeasible_truck_edges=infeasible_truck_edges,
        infeasible_drone_assign=infeasible_drone_assign,
    )
