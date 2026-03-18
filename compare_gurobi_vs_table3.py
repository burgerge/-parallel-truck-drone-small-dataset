"""
对比 gurobi_exact_small_enumeration 与 paper_table3_like 的结果
"""
import json
import os
from typing import Dict, List, Optional
import subprocess
import sys

def run_gurobi_enumeration(instance: str, p: int = 2) -> Dict:
    """运行 Gurobi 完全枚举"""
    print(f"\n{'='*80}")
    print(f"运行 Gurobi 完全枚举: {instance}, p={p}")
    print(f"{'='*80}")
    
    cmd = [
        sys.executable, "gurobi_exact_small_enumeration.py",
        "--instance", instance,
        "--p", str(p),
        "--time-limit", "300",
        "--quiet",
        "--out", f"outputs/compare_gurobi_{instance}_p{p}.json",
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"错误: {result.stderr}")
        return {}
    
    out_file = f"outputs/compare_gurobi_{instance}_p{p}.json"
    if os.path.exists(out_file):
        with open(out_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def run_table3_like(instance: str, p: int = 2, seed: int = 42) -> Dict:
    """运行 Table3-like (VNS + Gurobi)"""
    print(f"\n{'='*80}")
    print(f"运行 Table3-like: {instance}, p={p}, seed={seed}")
    print(f"{'='*80}")
    
    cmd = [
        sys.executable, "paper_table3_like.py",
        "--instances", instance,
        "--p", str(p),
        "--seed", str(seed),
        "--k-max", "7",
        "--l-max", "6",
        "--i-max", "5",
        "--paper-time-limit", "300",
        "--out-json", f"outputs/compare_table3_{instance}_p{p}_s{seed}.json",
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"错误: {result.stderr}")
        return {}
    
    out_file = f"outputs/compare_table3_{instance}_p{p}_s{seed}.json"
    if os.path.exists(out_file):
        with open(out_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def compare_results(gurobi_res: Dict, table3_res: Dict, instance: str, p: int):
    """对比结果"""
    print(f"\n{'='*80}")
    print(f"对比分析: {instance}, p={p}")
    print(f"{'='*80}\n")
    
    # 从 Gurobi 结果中获取最优解
    gurobi_best = None
    if "summary" in gurobi_res:
        # 单instance情况
        if gurobi_res["summary"].get("best_objective") is not None:
            gurobi_best = {
                "objective": float(gurobi_res["summary"]["best_objective"]),
                "open_depots": gurobi_res["summary"].get("best_open_depots", []),
                "open_depots_idx": gurobi_res["summary"].get("best_open_depots_idx", []),
                "status": "ENUMERATION (All combinations tested)",
            }
    
    # 从 Table3 结果中获取最优解
    table3_best = None
    if "table3_like" in table3_res:
        table3_data = table3_res["table3_like"]
        if isinstance(table3_data, dict) and "best_instance_seed_run" in table3_data:
            best_run = table3_data["best_instance_seed_run"]
            if best_run and "best_obj" in best_run:
                table3_best = {
                    "objective": float(best_run["best_obj"]),
                    "open_depots": best_run.get("best_x_open", []),
                    "open_depots_idx": best_run.get("best_x_open_idx", []),
                    "status": "VNS Search + Gurobi Validation",
                }
    
    print("=" * 80)
    print("1. 最优解对比")
    print("=" * 80)
    print(f"\nGurobi 完全枚举:")
    if gurobi_best:
        print(f"  目标函数值:      {gurobi_best['objective']:.6f}")
        print(f"  开仓数量:        {len(gurobi_best['open_depots'])}")
        print(f"  开仓位置(内部):  {gurobi_best['open_depots']}")
        print(f"  开仓位置(索引):  {gurobi_best['open_depots_idx']}")
        print(f"  方法说明:        {gurobi_best['status']}")
    else:
        print("  无有效结果")
    
    print(f"\nTable3-like (VNS + Gurobi):")
    if table3_best:
        print(f"  目标函数值:      {table3_best['objective']:.6f}")
        print(f"  开仓数量:        {len(table3_best['open_depots'])}")
        print(f"  开仓位置(内部):  {table3_best['open_depots']}")
        print(f"  开仓位置(索引):  {table3_best['open_depots_idx']}")
        print(f"  方法说明:        {table3_best['status']}")
    else:
        print("  无有效结果")
    
    if gurobi_best and table3_best:
        print(f"\n{'-'*80}")
        obj_diff = table3_best['objective'] - gurobi_best['objective']
        obj_pct_gap = (obj_diff / gurobi_best['objective']) * 100.0 if gurobi_best['objective'] != 0 else 0
        
        print(f"目标函数差距:")
        print(f"  绝对差值:        {obj_diff:+.6f}")
        print(f"  相对差值:        {obj_pct_gap:+.2f}%")
        print(f"  {' ' * 2}(负数表示 VNS 更优)")
        
        if gurobi_best['open_depots'] == table3_best['open_depots']:
            print(f"\n开仓策略: ✓ 一致")
        else:
            print(f"\n开仓策略: ✗ 不同")
            print(f"  Gurobi:  {gurobi_best['open_depots']}")
            print(f"  Table3:  {table3_best['open_depots']}")
    
    print("\n")
    
    # 详细路径对比
    print("=" * 80)
    print("2. 路径详情对比 (Gurobi 完全枚举最优解)")
    print("=" * 80)
    
    if gurobi_res and "combination_results" in gurobi_res:
        for combo in gurobi_res["combination_results"]:
            if combo.get("is_best") == 1:
                print(f"\n最优仓库组合: {combo.get('open_depots')}")
                print(f"  状态:        {combo.get('status_name')}")
                print(f"  目标函数值:  {combo.get('objective', 'N/A')}")
                print(f"  运行时间:    {combo.get('runtime_sec', 'N/A'):.2f}s")
                
                # 显示路径详情
                if "solution_details" in combo:
                    details = combo["solution_details"]
                    if "scenario_routes" in details:
                        print(f"\n  场景数量: {len(details['scenario_routes'])}")
                        for s_idx, scenario in enumerate(details['scenario_routes'][:1]):  # 只显示第一个场景
                            print(f"\n  场景 {s_idx}:")
                            for depot in scenario.get("depots", []):
                                depot_id = depot["depot_internal"]
                                truck_assigned = depot.get("truck_assigned_demands", [])
                                drone_assigned = depot.get("drone_assigned_demands", [])
                                truck_route = depot.get("truck_walk_eval_preferred", [])
                                drone_seq = depot.get("drone_sequence_spt", [])
                                drone_arrival = depot.get("drone_arrival_time", {})
                                
                                print(f"\n    仓库 {depot_id}:")
                                print(f"      卡车服务需求:  {truck_assigned}")
                                print(f"      无人机服务需求: {drone_assigned}")
                                print(f"      卡车路线:      {truck_route}")
                                if drone_seq:
                                    print(f"      无人机顺序:    {drone_seq}")
                                    print(f"      无人机到达时间: {drone_arrival}")
                else:
                    print("\n  (solution_details 为空 - 请检查return_solution_details参数)")
                break

def main():
    instance = "Instance1"
    p = 2
    seed = 42
    
    print(f"\n开始对比: {instance}, p={p}")
    
    # 运行两个方法
    gurobi_res = run_gurobi_enumeration(instance, p)
    table3_res = run_table3_like(instance, p, seed)
    
    # 对比结果
    if gurobi_res or table3_res:
        compare_results(gurobi_res, table3_res, instance, p)
    else:
        print("无法获取结果")

if __name__ == "__main__":
    main()
