
# main.py
# ============================================================
# 该文件提供两类能力：
# 1) 数据入口：load_instance_data 负责读取实例并构建 DataInstance
# 2) 轻量演示入口：main() 默认演示 A3（可手动切换到 A1）
#
# 说明：
# - 论文算法主体实现在 alg.py（A1~A5）
# - 本文件主要承担“把 txt 数据转为算法可用内存结构”的角色
# ============================================================
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
    单场景实例数据结构。

    字段说明：
    - num_nodes: 图中总节点数（需求点+可能的扩展depot节点）
    - demand_nodes: 需求点集合
    - adj: 邻接表（道路直接连接关系）
    - truck_times: 卡车行驶时间矩阵（不可达为 inf）
    - drone_times: 无人机飞行时间矩阵（通常为往返时间）
    - candidate_depots: 候选仓库节点（算法A1在此集合上选址）
    - drone_reachability: 每个depot可由无人机服务的需求点列表
    - depot_base_map: 内部depot节点ID -> W.txt原始ID 映射
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
    从实例目录读取 complete/t/v/W/D_k 并构建场景数据。

    文件角色：
    - complete.txt: 路网邻接矩阵（是否有直接边）
    - t.txt: 卡车时间（多个场景横向拼接）
    - v.txt: 无人机时间矩阵
    - W.txt: 候选仓库（原始ID）
    - D_k.txt: 每个候选仓库的无人机可达需求点（与W顺序对齐）

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
    # 这里兼容两种数据格式：
    # - depot 是 demand 的子集（同一ID空间）
    # - depot 被扩展成新节点（ID在 demand_count 之后）
    max_w = max(candidates) if candidates else -1
    max_dk = max((max(line) for line in dk_lines if line), default=-1)
    demand_count = max(max_w, max_dk) + 1
    if demand_count <= 0 or demand_count > num_nodes:
        demand_count = num_nodes
    if not dk_lines and demand_count < num_nodes:
        demand_count = num_nodes
    demand_nodes = list(range(demand_count))

    # ---- depot node mapping
    # 将内部depot节点ID映射回 W.txt 原始ID，便于汇报时与论文表格对齐。
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
    # t.txt 每行长度 = num_nodes * num_scenarios
    # 通过切片 [scen_idx*num_nodes : (scen_idx+1)*num_nodes] 还原每个场景的时间矩阵。
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
    """
    轻量调试入口：
    - 默认读取一个实例并演示 Algorithm 3 的构造输出；
    - 注释区可切换为 Algorithm 1 全流程。
    """
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
