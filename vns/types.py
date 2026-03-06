from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

Tour = List[int]
DroneSeq = List[int]
TTK = Dict[int, Tour]
DSK = Dict[int, DroneSeq]


@dataclass(frozen=True)
class Solution:
    tt: TTK
    ds: DSK


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    errors: List[str]
    missing: Set[int]
    duplicated: Set[int]
    infeasible_truck_edges: List[Tuple[int, int, int]]  # (depot, u, v)
    infeasible_drone_assign: List[Tuple[int, int]]  # (depot, demand)


@dataclass(frozen=True)
class RepairResult:
    solution: Optional[Solution]
    changed: bool
    notes: List[str]
