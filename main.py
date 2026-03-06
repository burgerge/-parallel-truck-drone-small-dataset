
# main.py
import os
import sys
from typing import List, Tuple

from alg import (
    HeuristicConfig,
    algorithm_1_stochastic_lrp,
    algorithm_2_depot_evaluation,
    algorithm_3_initial_construction,
    algorithm_4_find_backtrack,   # exposed mainly for testing
    algorithm_5_vns_improvement,  # exposed mainly for testing
    generate_scenarios_by_time_inflation
)

INSTANCES_ROOT = r"./Instance"
DEFAULT_INSTANCE_NAME = "Instance3"


# =========================
# Data Structure & Loader
# =========================
class DataInstance:
    """
    Stores one instance (one scenario): adjacency, truck/drone times, candidate depots, etc.
    """
    def __init__(self,
                 num_nodes,
                 demand_nodes,
                 adj_list,
                 truck_times,
                 drone_times,
                 candidate_depots,
                 drone_reachability,
                 depot_base_map):
        self.num_nodes = num_nodes
        self.demand_nodes = list(demand_nodes)
        self.adj = adj_list
        self.truck_times = truck_times
        self.drone_times = drone_times
        self.candidate_depots = list(candidate_depots)
        self.drone_reachability = drone_reachability
        # depot node id -> demand node id in W.txt
        self.depot_base_map = dict(depot_base_map)


def load_instance_data(folder_path: str) -> Tuple[DataInstance, List[DataInstance]]:
    """
    Reads: complete.txt, t.txt, v.txt, W.txt, D_k.txt from one instance folder.

    Returns:
    - base_instance: uses scenario 0 truck times
    - scenarios: list of DataInstance, one per scenario in t.txt
    """
    def load_matrix(filename):
        path = os.path.join(folder_path, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {filename}")
        matrix = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    row = [float(x) for x in line.replace(',', ' ').split()]
                    matrix.append(row)
        return matrix

    # 1) adjacency
    adj_matrix = load_matrix('complete.txt')
    num_nodes = len(adj_matrix)

    # 2) times
    raw_truck_times = load_matrix('t.txt')
    raw_drone_times = load_matrix('v.txt')

    # 3) candidate depots
    candidates = []
    w_path = os.path.join(folder_path, 'W.txt')
    if os.path.exists(w_path):
        with open(w_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    candidates.append(int(line.strip()))

    # 4) D_k (drone reachability) - order follows W.txt lines
    dk_lines: List[List[int]] = []
    dk_path = os.path.join(folder_path, 'D_k.txt')
    if os.path.exists(dk_path):
        with open(dk_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    parts = [int(x) for x in line.strip().replace(',', ' ').split()]
                    dk_lines.append(parts)

    # ---- demand node detection (README: 15 demand nodes, depots subset)
    max_w = max(candidates) if candidates else -1
    max_dk = max((max(line) for line in dk_lines if line), default=-1)
    demand_count = max(max_w, max_dk) + 1
    if demand_count <= 0 or demand_count > num_nodes:
        demand_count = num_nodes
    if not dk_lines and demand_count < num_nodes:
        demand_count = num_nodes
    demand_nodes = list(range(demand_count))

    # ---- depot node mapping
    if num_nodes == demand_count + len(candidates):
        depot_nodes = [demand_count + idx for idx in range(len(candidates))]
    else:
        depot_nodes = list(candidates)
    depot_base_map = {depot_nodes[i]: candidates[i] for i in range(min(len(depot_nodes), len(candidates)))}

    # ---- drone reachability keyed by depot node id
    drone_reachability = {}
    for idx, line in enumerate(dk_lines):
        if idx >= len(depot_nodes):
            break
        depot_node = depot_nodes[idx]
        reachable = [i for i in line if i in demand_nodes]
        drone_reachability[depot_node] = reachable

    # ---- preprocessing
    adj_list = {i: set() for i in range(num_nodes)}
    drone_times = {i: {} for i in range(num_nodes)}

    # v.txt: round-trip drone times
    for i in range(num_nodes):
        for j in range(num_nodes):
            drone_times[i][j] = raw_drone_times[i][j]
            if adj_matrix[i][j] == 1:
                adj_list[i].add(j)

    # ---- build scenarios from t.txt
    if not raw_truck_times:
        raise ValueError("t.txt is empty")
    row_len = len(raw_truck_times[0])
    if row_len % num_nodes != 0:
        raise ValueError("t.txt columns do not align with node count")
    num_scenarios = row_len // num_nodes
    scenarios: List[DataInstance] = []
    for scen_idx in range(num_scenarios):
        truck_times = {i: {} for i in range(num_nodes)}
        col_start = scen_idx * num_nodes
        col_end = col_start + num_nodes
        for i in range(num_nodes):
            row = raw_truck_times[i][col_start:col_end]
            for j in range(num_nodes):
                t = row[j]
                if adj_matrix[i][j] == 1 and t != 999:
                    truck_times[i][j] = t
                else:
                    truck_times[i][j] = 0 if i == j else float('inf')
        scenarios.append(
            DataInstance(num_nodes,
                         demand_nodes,
                         adj_list,
                         truck_times,
                         drone_times,
                         depot_nodes,
                         drone_reachability,
                         depot_base_map)
        )

    base_instance = scenarios[0] if scenarios else DataInstance(num_nodes,
                                                                demand_nodes,
                                                                adj_list,
                                                                {},
                                                                drone_times,
                                                                depot_nodes,
                                                                drone_reachability,
                                                                depot_base_map)
    return base_instance, scenarios


def main():
    # -------- instance selection
    instance_name = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INSTANCE_NAME
    folder = os.path.join(INSTANCES_ROOT, instance_name)
    if not os.path.exists(folder):
        print(f"[ERROR] Folder not found: {folder}")
        return

    instance, scenarios = load_instance_data(folder)
    print(f"Loaded: {instance_name}, nodes={instance.num_nodes}, candidates={instance.candidate_depots}")

    # -------- choose what to run
    # 1) Only Algorithm 3 (given depots)
    chosen_depots = set(instance.candidate_depots[:2]) if len(instance.candidate_depots) >= 2 else set(instance.candidate_depots)
    print(f"Run A3 with depots: {chosen_depots}")
    TTk, DSk = algorithm_3_initial_construction(chosen_depots, instance)
    for k in chosen_depots:
        print(f"\n[Depot {k}]")
        print("  Truck:", " -> ".join(map(str, TTk.get(k, []))))
        print("  Drone:", DSk.get(k, []))

    # 2) Full Algorithm 1 (swap local search) – uncomment to use
    # cfg = HeuristicConfig(num_depots_to_open=2, num_scenarios=10, seed=123, drone_time_is_roundtrip=True)
    # scen_list = scenarios if scenarios else None
    # best_X, best_f = algorithm_1_stochastic_lrp(instance, cfg=cfg, scenarios=scen_list)
    # print("\n=== Algorithm 1 result ===")
    # print("Open depots:", best_X)
    # print("Expected objective:", best_f)


if __name__ == "__main__":
    main()
