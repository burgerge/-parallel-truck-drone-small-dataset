from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

# 基础类型别名：
# - Tour: 某 depot 的卡车路径（节点序列）
# - DroneSeq: 某 depot 的无人机任务序列
Tour = List[int]
DroneSeq = List[int]
TTK = Dict[int, Tour]
DSK = Dict[int, DroneSeq]


@dataclass(frozen=True)
class Solution:
    """统一解结构：TT(卡车) + DS(无人机)。"""
    tt: TTK
    ds: DSK


@dataclass(frozen=True)
class ValidationResult:
    """可行性校验结果。"""
    ok: bool
    errors: List[str]
    missing: Set[int]
    duplicated: Set[int]
    infeasible_truck_edges: List[Tuple[int, int, int]]  # (depot, u, v)
    infeasible_drone_assign: List[Tuple[int, int]]  # (depot, demand)


@dataclass(frozen=True)
class RepairResult:
    """修复流程输出。"""
    solution: Optional[Solution]
    changed: bool
    notes: List[str]
