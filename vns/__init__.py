from vns.types import RepairResult, Solution, ValidationResult
from vns.vns_engine import improve, local_search, shaking

# vns 包对外暴露：
# - improve: Algorithm 5 的主入口
# - local_search/shaking: A5 内部关键步骤（便于测试和分析）
__all__ = [
    "Solution",
    "ValidationResult",
    "RepairResult",
    "shaking",
    "local_search",
    "improve",
]
