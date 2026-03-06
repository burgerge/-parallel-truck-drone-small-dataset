import argparse
import json
import os
import statistics
from typing import Dict, List, Optional, Sequence, Tuple

from alg import (
    HeuristicConfig,
    algorithm_3_initial_construction,
    algorithm_5_vns_improvement,
    calculate_arrival_times,
)
from main import load_instance_data
from vns.types import Solution
from vns.validate import validate_solution


def _safe_float(v, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _is_close(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol


def _normalized_weights(weights: Sequence[float], n: int) -> List[float]:
    if not weights:
        return [1.0 / n] * n
    s = float(sum(weights))
    if s <= 0:
        raise ValueError("Scenario weights must sum to positive value.")
    return [float(w) / s for w in weights]


def _pick_weights(report: Dict, n: int) -> List[float]:
    ws = report.get("scenario_weights")
    if isinstance(ws, list) and len(ws) == n:
        return _normalized_weights([_safe_float(x) for x in ws], n)

    details = report.get("best_seed_scenario_details")
    if isinstance(details, list) and len(details) == n:
        dws = [_safe_float(row.get("weight", 0.0)) for row in details]
        return _normalized_weights(dws, n)

    if report.get("mode") == "fixed-x":
        rows = report.get("scenarios")
        if isinstance(rows, list) and len(rows) == n:
            dws = [_safe_float(row.get("weight", 0.0)) for row in rows]
            return _normalized_weights(dws, n)
    return [1.0 / n] * n


def _cfg_from_report(report: Dict, seed: Optional[int]) -> HeuristicConfig:
    c = report.get("config", {})
    return HeuristicConfig(
        num_depots_to_open=int(c.get("num_depots_to_open", report.get("p", 2))),
        num_scenarios=int(c.get("num_scenarios", report.get("scenario_count", 10))),
        seed=seed,
        k_max=int(c.get("k_max", 7)),
        l_max=int(c.get("l_max", 6)),
        i_max=int(c.get("i_max", 20)),
        drone_time_is_roundtrip=bool(c.get("drone_time_is_roundtrip", True)),
        normalize_by_num_demands=bool(c.get("normalize_by_num_demands", True)),
        infeasible_penalty=float(c.get("infeasible_penalty", 1e9)),
    )


def _eval_scenario(x_open: set, scenario, cfg: HeuristicConfig) -> Dict:
    tt0, ds0 = algorithm_3_initial_construction(x_open, scenario)
    obj0 = calculate_arrival_times(tt0, ds0, scenario, cfg)
    tt1, ds1 = algorithm_5_vns_improvement(tt0, ds0, scenario, cfg)
    obj1 = calculate_arrival_times(tt1, ds1, scenario, cfg)
    vr = validate_solution(Solution(tt1, ds1), scenario)
    return {
        "obj_a3": obj0,
        "obj_a5": obj1,
        "improved": obj1 < obj0,
        "valid": vr.ok,
        "errors": vr.errors,
    }


def _audit_fixed_x(report: Dict, tol: float) -> Tuple[List[Dict], Dict]:
    checks: List[Dict] = []
    rows = report.get("scenarios", [])
    n = len(rows)
    if n == 0:
        checks.append({"name": "fixed-x has scenarios", "ok": False, "detail": "No scenarios in report"})
        return checks, {}

    weights = _pick_weights(report, n)
    calc_a3 = sum(weights[i] * _safe_float(rows[i].get("obj_a3")) for i in range(n))
    calc_a5 = sum(weights[i] * _safe_float(rows[i].get("obj_a5")) for i in range(n))
    rep_a3 = _safe_float(report.get("expected_obj_a3"))
    rep_a5 = _safe_float(report.get("expected_obj_a5"))

    checks.append(
        {
            "name": "expected_obj_a3 matches weighted sum",
            "ok": _is_close(calc_a3, rep_a3, tol),
            "detail": f"calc={calc_a3:.10f}, report={rep_a3:.10f}",
        }
    )
    checks.append(
        {
            "name": "expected_obj_a5 matches weighted sum",
            "ok": _is_close(calc_a5, rep_a5, tol),
            "detail": f"calc={calc_a5:.10f}, report={rep_a5:.10f}",
        }
    )

    implied = rep_a5 < rep_a3
    checks.append(
        {
            "name": "report improved flag consistent",
            "ok": bool(report.get("improved")) == implied,
            "detail": f"implied={implied}, report={report.get('improved')}",
        }
    )

    valid_count = sum(1 for r in rows if bool(r.get("valid")))
    improved_count = sum(1 for r in rows if bool(r.get("improved")))
    summary = {
        "scenario_count": n,
        "valid_count": valid_count,
        "improved_count": improved_count,
        "valid_rate": valid_count / n,
        "improved_rate": improved_count / n,
    }
    return checks, summary


def _audit_full_a1(report: Dict, tol: float) -> Tuple[List[Dict], Dict]:
    checks: List[Dict] = []
    runs = report.get("runs", [])
    n_runs = len(runs)
    if n_runs == 0:
        checks.append({"name": "full-a1 has runs", "ok": False, "detail": "No runs in report"})
        return checks, {}

    objs = [_safe_float(r.get("expected_obj")) for r in runs]
    rep_summary = report.get("summary", {})
    rep_best = _safe_float(rep_summary.get("best_expected_obj"))
    rep_avg = _safe_float(rep_summary.get("avg_expected_obj"))
    rep_std = _safe_float(rep_summary.get("std_expected_obj"))
    rep_best_seed = rep_summary.get("best_seed")
    rep_best_x = rep_summary.get("best_x_open")

    calc_best = min(objs)
    calc_avg = statistics.mean(objs)
    calc_std = statistics.pstdev(objs) if len(objs) > 1 else 0.0
    best_idx = min(range(len(runs)), key=lambda i: objs[i])
    calc_best_seed = runs[best_idx].get("seed")
    calc_best_x = runs[best_idx].get("best_x_open")

    checks.append(
        {
            "name": "summary.best_expected_obj consistent with runs",
            "ok": _is_close(rep_best, calc_best, tol),
            "detail": f"calc={calc_best:.10f}, report={rep_best:.10f}",
        }
    )
    checks.append(
        {
            "name": "summary.avg_expected_obj consistent with runs",
            "ok": _is_close(rep_avg, calc_avg, tol),
            "detail": f"calc={calc_avg:.10f}, report={rep_avg:.10f}",
        }
    )
    checks.append(
        {
            "name": "summary.std_expected_obj consistent with runs",
            "ok": _is_close(rep_std, calc_std, tol),
            "detail": f"calc={calc_std:.10f}, report={rep_std:.10f}",
        }
    )
    checks.append(
        {
            "name": "summary.best_seed consistent with runs",
            "ok": rep_best_seed == calc_best_seed,
            "detail": f"calc={calc_best_seed}, report={rep_best_seed}",
        }
    )
    checks.append(
        {
            "name": "summary.best_x_open consistent with runs",
            "ok": rep_best_x == calc_best_x,
            "detail": f"calc={calc_best_x}, report={rep_best_x}",
        }
    )
    checks.append(
        {
            "name": "all runtime_sec are non-negative",
            "ok": all(_safe_float(r.get("runtime_sec"), -1.0) >= 0.0 for r in runs),
            "detail": "runtime check",
        }
    )

    details = report.get("best_seed_scenario_details", [])
    summary = {"run_count": n_runs}
    if isinstance(details, list) and details:
        n = len(details)
        weights = _pick_weights(report, n)
        calc_from_details = sum(weights[i] * _safe_float(details[i].get("obj_a5")) for i in range(n))
        valid_count = sum(1 for r in details if bool(r.get("valid")))
        improved_count = sum(1 for r in details if bool(r.get("improved")))

        checks.append(
            {
                "name": "best_seed expected objective consistent with scenario details",
                "ok": _is_close(calc_from_details, rep_best, tol),
                "detail": f"calc={calc_from_details:.10f}, report={rep_best:.10f}",
            }
        )

        summary.update(
            {
                "scenario_count": n,
                "valid_count": valid_count,
                "improved_count": improved_count,
                "valid_rate": valid_count / n,
                "improved_rate": improved_count / n,
            }
        )
    return checks, summary


def _recompute_best_seed(
    report: Dict, instances_root: str, tol: float
) -> Tuple[List[Dict], Dict]:
    checks: List[Dict] = []
    mode = report.get("mode")
    if mode != "full-a1":
        checks.append({"name": "recompute supported for full-a1 only", "ok": True, "detail": f"mode={mode}"})
        return checks, {}

    instance_name = report.get("instance")
    if not instance_name:
        checks.append({"name": "instance present", "ok": False, "detail": "Missing report.instance"})
        return checks, {}

    folder = os.path.join(instances_root, instance_name)
    if not os.path.exists(folder):
        checks.append({"name": "instance folder exists", "ok": False, "detail": folder})
        return checks, {}

    _, scenarios = load_instance_data(folder)
    if not scenarios:
        checks.append({"name": "scenarios loaded", "ok": False, "detail": "No scenarios loaded"})
        return checks, {}

    summary = report.get("summary", {})
    best_seed = summary.get("best_seed")
    x_open = summary.get("best_x_open", [])
    if best_seed is None or not isinstance(x_open, list) or not x_open:
        checks.append(
            {"name": "best_seed and best_x_open present", "ok": False, "detail": "Missing best seed or best x_open"}
        )
        return checks, {}

    weights = _pick_weights(report, len(scenarios))
    cfg = _cfg_from_report(report, seed=int(best_seed))

    detail_rows = []
    expected = 0.0
    x_set = set(int(x) for x in x_open)
    for i, scen in enumerate(scenarios):
        row = _eval_scenario(x_set, scen, cfg)
        detail_rows.append(row)
        expected += weights[i] * _safe_float(row["obj_a5"])

    rep_best = _safe_float(summary.get("best_expected_obj"))
    checks.append(
        {
            "name": "recomputed best seed expected objective matches report",
            "ok": _is_close(expected, rep_best, tol),
            "detail": f"recomputed={expected:.10f}, report={rep_best:.10f}",
        }
    )
    valid_count = sum(1 for r in detail_rows if bool(r["valid"]))
    improved_count = sum(1 for r in detail_rows if bool(r["improved"]))
    recompute_summary = {
        "scenario_count": len(detail_rows),
        "valid_count": valid_count,
        "improved_count": improved_count,
        "valid_rate": valid_count / len(detail_rows),
        "improved_rate": improved_count / len(detail_rows),
        "recomputed_expected_obj": expected,
    }
    return checks, recompute_summary


def _print_checks(title: str, checks: List[Dict]) -> None:
    print(f"\n[{title}]")
    for c in checks:
        status = "PASS" if c["ok"] else "FAIL"
        print(f"- {status}: {c['name']} | {c['detail']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit experiment output JSON for internal consistency.")
    parser.add_argument("--report", required=True, help="Path to report JSON generated by run_experiment.py")
    parser.add_argument("--instances-root", default="./Instance")
    parser.add_argument("--tol", type=float, default=1e-6, help="Tolerance for float consistency checks")
    parser.add_argument("--recompute-best-seed", action="store_true")
    parser.add_argument("--out", default="", help="Optional path to save audit report JSON")
    parser.add_argument("--strict", action="store_true", help="Exit with code 1 if any check fails")
    args = parser.parse_args()

    with open(args.report, "r", encoding="utf-8") as f:
        report = json.load(f)

    mode = report.get("mode")
    if mode not in {"fixed-x", "full-a1"}:
        raise ValueError(f"Unsupported report mode: {mode}")

    if mode == "fixed-x":
        checks, summary = _audit_fixed_x(report, args.tol)
    else:
        checks, summary = _audit_full_a1(report, args.tol)

    recompute_checks = []
    recompute_summary = {}
    if args.recompute_best_seed:
        recompute_checks, recompute_summary = _recompute_best_seed(report, args.instances_root, args.tol)

    _print_checks("Consistency Checks", checks)
    if recompute_checks:
        _print_checks("Recompute Checks", recompute_checks)

    all_checks = checks + recompute_checks
    failed = [c for c in all_checks if not c["ok"]]
    passed = [c for c in all_checks if c["ok"]]

    print("\n[Summary]")
    print(f"- report: {args.report}")
    print(f"- mode: {mode}")
    print(f"- passed_checks: {len(passed)}")
    print(f"- failed_checks: {len(failed)}")
    if summary:
        print(f"- audit_metrics: {summary}")
    if recompute_summary:
        print(f"- recompute_metrics: {recompute_summary}")

    out = {
        "report_path": args.report,
        "mode": mode,
        "checks": all_checks,
        "summary": summary,
        "recompute_summary": recompute_summary,
        "passed_checks": len(passed),
        "failed_checks": len(failed),
    }
    if args.out:
        out_dir = os.path.dirname(args.out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"- audit_saved_to: {args.out}")

    if args.strict and failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
