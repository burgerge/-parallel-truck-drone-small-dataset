import argparse

from paper_eval_common import build_cfg, parse_instance_names, save_json
from paper_table3_like import run_table3_like, save_table3_csv
from paper_table4_like import run_table4_like, save_table4_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Run both Table3-like and Table4-like experiments.")
    parser.add_argument("--instances-root", default="./Instance")
    parser.add_argument("--instances", default="1-10", help="Examples: 1-10 | 1,2,3 | Instance3,Instance4")
    parser.add_argument("--p", type=int, default=2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--k-max", type=int, default=7)
    parser.add_argument("--l-max", type=int, default=6)
    parser.add_argument("--i-max", type=int, default=5)
    parser.add_argument("--out-json", default="outputs/table3_table4_like_current_code.json")
    parser.add_argument("--out-table3-csv", default="outputs/table3_like_current_code.csv")
    parser.add_argument("--out-table4-csv", default="outputs/table4_like_current_code.csv")
    args = parser.parse_args()

    instance_names = parse_instance_names(args.instances)
    cfg = build_cfg(args.p, args.seed, args.k_max, args.l_max, args.i_max)

    table3 = run_table3_like(args.instances_root, instance_names, cfg)
    table4 = run_table4_like(args.instances_root, instance_names, cfg)

    payload = {
        "note": "Table3/Table4-like reproduction with current code (no Gurobi columns).",
        "config": {
            "instances": instance_names,
            "p": args.p,
            "seed": args.seed,
            "k_max": args.k_max,
            "l_max": args.l_max,
            "i_max": args.i_max,
        },
        "table3_like": table3,
        "table4_like": table4,
    }
    save_json(args.out_json, payload)
    save_table3_csv(args.out_table3_csv, table3)
    save_table4_csv(args.out_table4_csv, table4)

    print(f"Saved JSON: {args.out_json}")
    print(f"Saved CSV : {args.out_table3_csv}")
    print(f"Saved CSV : {args.out_table4_csv}")


if __name__ == "__main__":
    main()
