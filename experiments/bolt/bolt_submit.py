"""
Submit experiments to Apple Bolt.

Usage:
    python ./bolt/bolt_submit.py 0 1 2             # submit experiments with IDs 0, 1, 2
    python ./bolt/bolt_submit.py all               # submit all experiments
    python ./bolt/bolt_submit.py 0 1 --dry-run     # print commands without submitting
    python ./bolt/bolt_submit.py all --workers 8   # submit in parallel with 8 workers (default)
"""

import subprocess
import argparse
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

_print_lock = threading.Lock()


def load_experiments(experiments_file):
    """Parse bolt_exps.jsonl into {id: command}. Lines starting with '//' are comments."""
    experiments = {}
    with open(experiments_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            exp_id, _, command = line.partition(":")
            experiments[exp_id.strip()] = command.strip()
    return experiments


def submit(config_template, exp_id, command, dry_run=False):
    os.makedirs("tmp", exist_ok=True)

    with open(config_template) as f:
        content = f.read()

    content = content.replace("COMMAND_PLACEHOLDER", command)
    content = content.replace("Progressive-RL", f"Progressive-RL-{exp_id}")

    tmp_config = f"bolt/tmp/config_{exp_id}.yaml"
    with open(tmp_config, "w") as f:
        f.write(content)

    submit_cmd = f"bolt task submit --tar . --config {tmp_config}"

    lines = [f"[{exp_id}] {command}", f"       (pwd) {os.getcwd()}"]
    if dry_run:
        lines.append(f"       (dry run) {submit_cmd}")
        with _print_lock:
            print("\n".join(lines) + "\n", flush=True)
        return

    lines.append(f"       {submit_cmd}")
    result = subprocess.run(submit_cmd, shell=True, check=True, capture_output=True, text=True)
    if result.stdout:
        lines.append(result.stdout.rstrip())
    if result.stderr:
        lines.append(result.stderr.rstrip())
    with _print_lock:
        print("\n".join(lines) + "\n", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Submit experiments to Apple Bolt.")
    parser.add_argument("ids", nargs="+", help='Experiment IDs to submit, or "all"')
    parser.add_argument(
        "--config",
        default="./bolt/configs/iris_A100_1node_new.yaml",
        help="Bolt YAML config template",
    )
    parser.add_argument(
        "--experiments",
        default="./bolt/bolt_exps.jsonl",
        help="Experiments file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without submitting",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel submissions (default: 8)",
    )
    args = parser.parse_args()

    experiments = load_experiments(args.experiments)

    ids_to_run = list(experiments.keys()) if args.ids == ["all"] else args.ids

    valid = [(eid, experiments[eid]) for eid in ids_to_run if eid in experiments]
    for eid in ids_to_run:
        if eid not in experiments:
            print(f"Warning: ID '{eid}' not found in {args.experiments}", file=sys.stderr)

    def submit_one(eid, cmd):
        submit(args.config, eid, cmd, args.dry_run)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(submit_one, eid, cmd): eid for eid, cmd in valid}
        for future in as_completed(futures):
            future.result()  # re-raise any exceptions


if __name__ == "__main__":
    main()
