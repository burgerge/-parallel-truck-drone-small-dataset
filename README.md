# Parallel Truck-Drone Small Instances

本仓库用于复现并扩展论文中的并行卡车-无人机两阶段选址与路径问题（小规模实例）。

当前代码包含两类方法：
- 启发式主线（A1-A5 + VNS）
- Gurobi 精确/半精确求解（含开仓组合穷举版本）

---

## 1. 目录与数据

- `Instance/Instance1..10/`：算例数据目录（`complete.txt`, `t.txt`, `v.txt`, `W.txt`, `D_k.txt`）。
- `outputs/`：所有实验输出（CSV/JSON）。
- `vns/`：VNS 子模块（邻域、修复、可行性校验、目标计算等）。

---

## 2. 运行入口（按任务）

- 论文 Table3-like：`paper_table3_like.py`
- 论文 Table4-like：`paper_table4_like.py`
- 单实例/多 seed 实验：`run_experiment.py`
- 报告一致性审计：`audit_run.py`
- Gurobi 精确两阶段模型：`gurobi_exact_small.py`
- Gurobi 穷举开仓组合 + 二阶段求解：`gurobi_exact_small_enumeration.py`

---

## 3. 每个 Python 文件功能说明

### 根目录文件

`alg.py`
- 启发式总入口，实现论文算法 A1-A5 主链。
- 关键函数：
  - `algorithm_1_stochastic_lrp`：开仓集合交换搜索（A1）
  - `algorithm_2_depot_evaluation`：跨场景期望评估（A2）
  - `algorithm_3_initial_construction`：初始卡车/无人机构造（A3）
  - `algorithm_4_find_backtrack`：回溯路径子过程（A4）
  - `algorithm_5_vns_improvement`：VNS 改进入口（A5）
  - `calculate_arrival_times`：目标函数（平均/总到达时间）
- `HeuristicConfig` 包含算法参数（`p`, `k_max`, `l_max`, `i_max` 等）及 `strict_feasibility`（默认 `True`）。

`main.py`
- 数据加载核心：`load_instance_data(folder)`。
- 把文本文件解析成 `DataInstance` 结构，并生成场景列表。
- 自带轻量 demo（默认演示 A3，可手动切到 A1 全流程）。

`run_experiment.py`
- 通用实验驱动脚本。
- 模式：
  - `fixed-x`：固定开仓组合评估
  - `full-a1`：运行 A1（可多 seed）
- 输出场景级和汇总级 JSON，支持 `--include-best-seed-details`。

`audit_run.py`
- 审计 `run_experiment.py` 输出 JSON 的内部一致性。
- 检查内容包括：加权期望是否一致、summary 与 runs 是否一致、可选重算 best seed。

`paper_eval_common.py`
- `paper_table3_like.py` / `paper_table4_like.py` 的公共工具模块。
- 提供实例选择解析、配置构建、A1 执行封装、CI95 计算、JSON 保存等。

`paper_table3_like.py`
- 复现实验 Table3-like（with/without improvement）。
- 产出：
  - 行级结果（每个实例）
  - 平均指标
  - CSV/JSON
- `with_open_depots` / `without_open_depots` 使用 `0-4` 候选仓编号（`W` 顺序索引）。

`paper_table4_like.py`
- 复现实验 Table4-like（逐个禁用邻域 N1..N7）。
- 相对 baseline 统计目标增幅、CPU 降幅及 CI95。
- 输出 CSV/JSON。

`gurobi_exact_small.py`
- 一体化两阶段 MILP（含选址变量 `x`）。
- 更接近“单模型联合求解”，但规模敏感，可能慢。
- 支持时间上限、MIP gap、求解参数等。

`gurobi_exact_small_enumeration.py`
- 先枚举开仓组合，再对每个组合单独求二阶段模型。
- 可输出：
  - 组合明细 CSV（每个组合一行）
  - 每实例最优 CSV（每实例一行）
  - 汇总 JSON（可批量 `Instance1-10`）
- `open_depots_idx` 同样是 `0-4` 索引。

### `vns/` 子模块

`vns/__init__.py`
- 包初始化文件。

`vns/types.py`
- 数据结构定义：
  - `Solution`（`tt` + `ds`）
  - `ValidationResult`
  - `RepairResult`

`vns/objective.py`
- VNS 内部统一目标函数接口。
- 实际复用 `alg.calculate_arrival_times`，确保口径一致。

`vns/validate.py`
- 解可行性检查：
  - 卡车断边
  - 无人机不可达服务
  - 需求点重复/遗漏服务

`vns/repair.py`
- 可行性修复器：
  - 清理非法服务
  - 去重
  - 补齐遗漏需求
  - 断边最短路拼接修复

`vns/truck_ops.py`
- 卡车路径基础操作：
  - 可行边判断
  - 最优插入位置
  - 最短路（Dijkstra）

`vns/drone_ops.py`
- 无人机辅助操作：
  - 可达性检查
  - 往返时间读取
  - 序列排序（按飞行时间）

`vns/neighborhoods.py`
- VNS 邻域定义 N1..N7（卡车/无人机跨仓或同仓重分配、重构等）。
- `NEIGHBORHOODS` 字典是邻域编号到函数的映射。

`vns/vns_engine.py`
- VNS 总控：
  - `shaking`
  - `local_search`
  - `improve`
- 当前版本支持严格可行性模式：
  - 邻域候选先修复再评分
  - 接受解前进行可行性门控（由 `strict_feasibility` 控制）

---

## 4. 当前启发式“严格可行”策略

默认 `strict_feasibility=True`。执行逻辑：
- VNS 候选比较前先尝试 `repair`
- A2 场景评估前做 `validate/repair` 兜底
- 修复失败场景按 `infeasible_penalty` 处理

目的：避免“重复服务但目标看起来更小”的偏乐观结果。

---

## 5. 常用命令

运行 Table3-like：

```bash
python paper_table3_like.py --instances 1-10 --out-json outputs/table3_like_current_code.json --out-csv outputs/table3_like_current_code.csv
```

运行 Table4-like：

```bash
python paper_table4_like.py --instances 1-10 --out-json outputs/table4_like_current_code.json --out-csv outputs/table4_like_current_code.csv
```

批量跑 Gurobi 穷举版并汇总：

```bash
python gurobi_exact_small_enumeration.py --instances 1-10 --p 2 --time-limit 1800 --mip-gap 0 --out outputs/gurobi_enum_all_instances.json --out-csv outputs/gurobi_enum_all_instances_combos.csv --out-best-csv outputs/gurobi_enum_all_instances_best.csv --quiet --continue-on-error
```

---

## 6. 输出文件约定

- `*_combos.csv`：每个开仓组合一行（含 `open_depots_idx`）。
- `*_best.csv`：每个实例一行最优结果。
- `*.json`：完整结构化结果，可用于后续审计与复现实验。

---

## 7. 当前对比结果文件与含义

当前这批“仓库 Table3 路径 vs Gurobi 重跑结果”的主要文件在：

- `outputs/gurobi_enum_paper_exact_100s_20260318_195048/`
- `outputs/gurobi_enum_paper_exact_100s_20260318_195048/comparison_bundle_vs_table3/`

其中 `gurobi_enum_paper_exact_100s_20260318_195048` 表示：
- Gurobi 使用 `gurobi_exact_small_enumeration.py`
- `Instance1-10`
- `p=2`
- 每个组合 `100s`
- 每个实例共 `10` 个组合，理论总预算约 `1000s`

### 7.1 最重要的几个文件

`outputs/gurobi_enum_paper_exact_100s_20260318_195048/gurobi_enum_all_instances_paper_exact_100s_combos.csv`
- Gurobi 穷举结果明细。
- 每个实例的每个开仓组合一行。
- 用来看：某个实例所有组合的 `objective / best_bound / runtime / status`。

`outputs/gurobi_enum_paper_exact_100s_20260318_195048/gurobi_enum_all_instances_paper_exact_100s_best.csv`
- Gurobi 穷举结果摘要。
- 每个实例只保留当前 best combo 一行。

`outputs/gurobi_enum_paper_exact_100s_20260318_195048/comparison_bundle_vs_table3/comparison_summary.csv`
- 当前最核心的实例级对照表。
- 每个实例一行。
- 用来看：
  - Table3 当前仓库路径对应的开仓组合
  - Gurobi 当前 best combo
  - 两边在 `paper_exact` 口径下的目标值
  - 两边组合是否一致
  - Gurobi 当前状态是否是 `TIME_LIMIT`/`OPTIMAL`

`outputs/gurobi_enum_paper_exact_100s_20260318_195048/comparison_bundle_vs_table3/detailed_instance_comparison.csv`
- 更详细的实例级对照表。
- 除了目标值和时间，还保留：
  - `table3_paths_raw_0_4_json`：原始 Table3 路径
  - `table3_paths_0_4_json`：修复成 `paper_exact` 可评估后的 Table3 路径
  - `gurobi_paths_0_4_json`：Gurobi 最优组合导出的路径

`outputs/gurobi_enum_paper_exact_100s_20260318_195048/comparison_bundle_vs_table3/detailed_route_comparison.csv`
- 路径级对照表。
- 每个 `instance / scenario / depot` 一行。
- 用来看具体某个场景、某个 depot 下：
  - Table3 路径怎么走
  - Gurobi 路径怎么走
  - 无人机序列是否不同

`outputs/gurobi_enum_paper_exact_100s_20260318_195048/comparison_bundle_vs_table3/paper_report_comparison_compact.csv`
- 汇报版精简表。
- 只保留：
  - `instance`
  - `table3_combo`
  - `gurobi_combo`
  - `table3_obj`
  - `gurobi_obj`
  - `gap_table3_minus_gurobi`
  - `table3_time_sec`
  - `gurobi_time_sec`

`outputs/gurobi_enum_paper_exact_100s_20260318_195048/comparison_bundle_vs_table3/route_bundle_index.csv`
- 路径 JSON 索引表。
- 用来定位每个实例对应的：
  - 原始 Table3 JSON
  - repaired Table3 JSON
  - Gurobi best JSON

### 7.2 raw Table3 与 repaired Table3 的区别

仓库里原始 Table3 路径主要来自：

- `outputs/routes_bundle_10instances/Instance*_table3_fixedx_combo_*.json`

这些文件是启发式导出的显式路径，但它们并不总能直接被 `paper_exact` 模型无损接受，原因包括：
- 卡车路径没有显式写回仓
- 开着的 depot 在某些场景下卡车完全不动
- 同一条 truck arc 被重复走过，但 exact 模型里的 `y` 是二进制

因此当前比较流程会先生成 repaired 版本：

- `outputs/gurobi_enum_paper_exact_100s_20260318_195048/comparison_bundle_vs_table3/Instance*_table3_fixedx_combo_*_paper_exact_repaired.json`

这些 repaired JSON 的作用是：
- 尽量保留原始 Table3 路径语义
- 把路径修成 `paper_exact` 模型可评估的形式
- 让 `table3_obj` 真正有可比性

特别是 `Instance6` 和 `Instance10`：
- 原始 Table3 显式路径在严格 `paper_exact` 固定下曾经是 `INFEASIBLE`
- 现在已经通过 repaired 版本补成 `OPTIMAL` 可评估结果

### 7.3 关键字段怎么理解

#### depot 编号口径

- `*_combo_idx` 或路径中的 `D0` 到 `D4`：表示候选仓编号，按 `W` 的顺序索引，口径是 `0-4`
- `*_combo_internal` 或 `depot_internal`：表示内部节点编号，当前实例里通常是 `15-19`

因此：
- `table3_combo = [0, 1]` 表示选的是候选仓 0 和 1
- 不等于内部节点 0 和 1

#### 目标值字段

`table3_repo_final_obj`
- 当前仓库启发式流程自己的最终评价值。
- 它保留作参考，但**不应直接拿来和 Gurobi 的 `paper_exact` 目标值作严格对比**。

`table3_fixed_path_paper_obj` / `table3_obj`
- 把 Table3 路径修成 `paper_exact` 可行结构后，再放进 `solve_fixed_combo_commodity(...)` 重评得到的目标值。
- 这是当前与 Gurobi 进行公平比较时，Table3 一侧应使用的目标值。

`gurobi_paper_obj` / `paper_obj_gurobi_best_combo_opt` / `gurobi_obj`
- Gurobi 枚举结果里当前 best combo 的 `paper_exact` 目标值。

`gap_table3_minus_gurobi`
- 计算方式是 `table3_obj - gurobi_obj`
- 大于 `0`：Gurobi 更好
- 小于 `0`：当前 repaired Table3 路径更好

#### 时间字段

`table3_time_sec`
- 来自当前仓库 `table3_like_paper_exact_unified.csv` 的启发式运行时间。
- 这是仓库当前代码的时间，不是论文 PDF 原始 Table 3 的 CPU 时间。

`gurobi_time_sec`
- 来自当前 Gurobi best combo 那一行的求解时间。
- 它通常接近 `100s`，因为本轮设置了 `100s/combo`。
- 这不是“整个实例枚举完 10 个组合”的总时间，而是当前 best combo 那一次求解的时间。

#### 状态字段

`paper_status_table3_path_fixed`
- repaired Table3 路径在严格 `paper_exact` 模型下的状态。

`gurobi_best_status`
- Gurobi 当前 best combo 的状态。
- 若为 `TIME_LIMIT`，表示：
  - 已有 incumbent 目标值
  - 但当前 `100s/combo` 下尚未证明该组合已达到全局最优

### 7.4 当前推荐阅读顺序

如果只想快速看最终对比：
- 先看 `paper_report_comparison_compact.csv`

如果想看实例级完整信息：
- 看 `comparison_summary.csv`
- 再看 `detailed_instance_comparison.csv`

如果想追踪某个实例的具体路径差别：
- 先看 `route_bundle_index.csv`
- 再开对应的 repaired Table3 JSON 与 Gurobi best JSON
- 或直接看 `detailed_route_comparison.csv`

### 7.5 这些文件是怎么生成的

主要由以下脚本生成：

- `gurobi_exact_small_enumeration.py`
  - 负责重跑 Gurobi 枚举结果

- `export_enumeration_vs_table3_bundle.py`
  - 负责把 Gurobi best 结果与 Table3 路径组织成同一套 comparison bundle
  - 同时导出 repaired Table3 JSON

- `build_detailed_table3_gurobi_comparison.py`
  - 负责生成详细实例表、详细路径表和汇报版精简表

如果需要重刷当前这一套对比文件，通常按下面顺序：

```bash
python gurobi_exact_small_enumeration.py ...
python export_enumeration_vs_table3_bundle.py ...
python build_detailed_table3_gurobi_comparison.py ...
```

### 7.6 当前这批对比文件的运行链路

目前这批结果**不是由单个 all-in-one Python 脚本一次性跑完的**，而是分步生成的。

#### A. 当前推荐链路：重现 `comparison_bundle_vs_table3`

如果你已经有：
- `outputs/routes_bundle_10instances/` 里的 Table3 路径 JSON
- `outputs/table3_like_paper_exact_unified.csv` 这个 Table3 摘要表

那么当前推荐链路是 **3 个脚本顺序运行**：

**第 1 步：跑 Gurobi 枚举**

```powershell
python gurobi_exact_small_enumeration.py --instances 1-10 --p 2 --time-limit 100 --mip-gap 0 --mip-focus 1 --presolve 2 --cuts 2 --symmetry 2 --out outputs/gurobi_enum_paper_exact_100s_20260318_195048/gurobi_enum_all_instances_paper_exact_100s.json --out-csv outputs/gurobi_enum_paper_exact_100s_20260318_195048/gurobi_enum_all_instances_paper_exact_100s_combos.csv --out-best-csv outputs/gurobi_enum_paper_exact_100s_20260318_195048/gurobi_enum_all_instances_paper_exact_100s_best.csv --per-instance-dir outputs/gurobi_enum_paper_exact_100s_20260318_195048/per_instance --quiet --continue-on-error
```

这一步产出：
- `gurobi_enum_all_instances_paper_exact_100s.json`
- `gurobi_enum_all_instances_paper_exact_100s_combos.csv`
- `gurobi_enum_all_instances_paper_exact_100s_best.csv`
- `per_instance/Instance*.json`

**第 2 步：把 Gurobi 结果和现有 Table3 路径绑成 comparison bundle**

```powershell
python export_enumeration_vs_table3_bundle.py --enum-per-instance-dir outputs/gurobi_enum_paper_exact_100s_20260318_195048/per_instance --table3-bundle-dir outputs/routes_bundle_10instances --instances-root ./Instance --instances 1-10 --time-limit 100 --mip-gap 0 --threads 0 --mip-focus 1 --presolve 2 --cuts 2 --symmetry 2 --out-dir outputs/gurobi_enum_paper_exact_100s_20260318_195048/comparison_bundle_vs_table3
```

这一步产出：
- `comparison_summary.csv/json`
- `route_bundle_index.csv/json`
- `gurobi_best_paths_flat.csv`
- `table3_paths_flat.csv`
- `Instance*_gurobi_best_combo_*.json`
- `Instance*_table3_fixedx_combo_*_paper_exact_repaired.json`

**第 3 步：生成详细表和汇报版精简表**

```powershell
python build_detailed_table3_gurobi_comparison.py --comparison-summary-csv outputs/gurobi_enum_paper_exact_100s_20260318_195048/comparison_bundle_vs_table3/comparison_summary.csv --table3-unified-csv outputs/table3_like_paper_exact_unified.csv --instances-root ./Instance --out-dir outputs/gurobi_enum_paper_exact_100s_20260318_195048/comparison_bundle_vs_table3
```

这一步产出：
- `detailed_instance_comparison.csv/json`
- `detailed_route_comparison.csv`

然后再从 `detailed_instance_comparison.csv` 裁出汇报版：

```powershell
@'
import csv
from pathlib import Path
src = Path("outputs/gurobi_enum_paper_exact_100s_20260318_195048/comparison_bundle_vs_table3/detailed_instance_comparison.csv")
out = Path("outputs/gurobi_enum_paper_exact_100s_20260318_195048/comparison_bundle_vs_table3/paper_report_comparison_compact.csv")
cols = [
    ("instance", "instance"),
    ("table3_combo_idx", "table3_combo"),
    ("gurobi_combo_idx", "gurobi_combo"),
    ("table3_fixed_path_paper_obj", "table3_obj"),
    ("gurobi_paper_obj", "gurobi_obj"),
    ("obj_gap_table3_minus_gurobi", "gap_table3_minus_gurobi"),
    ("table3_repo_runtime_sec", "table3_time_sec"),
    ("gurobi_runtime_sec", "gurobi_time_sec"),
]
with src.open("r", encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))
with out.open("w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=[dst for _, dst in cols])
    w.writeheader()
    for row in rows:
        w.writerow({dst: row.get(src_name, "") for src_name, dst in cols})
'@ | python -
```

#### B. Table3 侧输入文件分别从哪里来

当前对比链路依赖两类 Table3 侧输入，它们不是由 `export_enumeration_vs_table3_bundle.py` 现场生成的，而是预先存在：

**1. `outputs/table3_like_paper_exact_unified.csv/json`**
- 由 `paper_table3_like.py` 生成。
- 它给出 Table3-like with/without improvement 的实例级摘要。
- 在当前对比表里主要提供：
  - `table3_time_sec`
  - `table3_repo_final_obj`

典型命令：

```powershell
python paper_table3_like.py --instances 1-10 --p 2 --seed 123 --k-max 7 --l-max 6 --i-max 5 --strict-feasibility true --final-metric paper_exact --paper-time-limit 100 --paper-mip-gap 0 --paper-mip-focus 1 --paper-presolve 2 --paper-cuts 2 --paper-symmetry 2 --out-json outputs/table3_like_paper_exact_unified.json --out-csv outputs/table3_like_paper_exact_unified.csv
```

**2. `outputs/routes_bundle_10instances/Instance*_table3_fixedx_combo_*.json`**
- 这是 Table3 路径级输入。
- 它们是历史生成的 route bundle，当前比较直接复用它们。
- 在当前对比链路里主要提供：
  - 原始 Table3 显式 truck/drone 路径
  - `expected_obj_a5`

这些文件最初由 `export_routes_bundle_10instances.py` 导出，但它本身依赖一个中间表：
- `outputs/compare_table3_paper_unified_vs_gurobi.csv`

也就是说，**从零重建 `routes_bundle_10instances` 目前不是一条最简洁的新链路**，而是旧流程留下的中间产物。当前仓库里没有一个“一键脚本”把：
- `paper_table3_like.py`
- `routes_bundle_10instances`
- `gurobi_exact_small_enumeration.py`
- `comparison_bundle_vs_table3`

全部一步串起来。

#### C. 结论：目前是分步跑，不是一个脚本全包

因此当前答案是：

- **有清晰的分步链路**
- **没有单个 Python 文件把当前这套对比结果一键全部生成**

如果只重现“当前这批对比表”，推荐用上面的 **A 链路**。
如果连 Table3 侧输入也要一起重建，则还需要 **B 链路**里的历史输入步骤。
