#!/usr/bin/env python3
"""Run a command under a total-system memory watchdog.

Wraps a command in its own process group, samples system-wide memory every
interval, and kills the group before the host runs out of memory. Writes a
memory trace and, on a trip, a reason file naming the threshold that fired.

Standard library only, so it runs under any Python and cannot itself fail on a
missing dependency.

    scripts/memguard.py -- uv run jupyter nbconvert --execute nb.ipynb
    scripts/memguard.py --min-free 64 -- uv run python train.py

Exit codes:
    0-N  the wrapped command's own exit code
    137  the watchdog killed the command (out of memory)
"""

from __future__ import annotations

import argparse
import csv
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

GIB = 1024**3


def page_size() -> int:
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    first = out.splitlines()[0]
    for token in first.replace(")", " ").split():
        if token.isdigit():
            return int(token)
    return 16384


def memory_snapshot(pagesize: int) -> dict[str, float]:
    """Total-system memory in GiB.

    `available` follows the macOS convention: pages that can be handed to a new
    allocation without paging out real work, which is free plus inactive plus
    speculative plus purgeable.
    """
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    pages: dict[str, int] = {}
    for line in out.splitlines()[1:]:
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        digits = value.strip().rstrip(".")
        if digits.isdigit():
            pages[key.strip()] = int(digits)

    def gib(*names: str) -> float:
        return sum(pages.get(name, 0) for name in names) * pagesize / GIB

    available = gib(
        "Pages free", "Pages inactive", "Pages speculative", "Pages purgeable"
    )

    swap = subprocess.run(
        ["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True
    ).stdout
    swap_used = 0.0
    tokens = swap.split()
    for index, token in enumerate(tokens):
        if token == "used" and index + 2 < len(tokens):
            raw = tokens[index + 2]
            scale = {"M": 1 / 1024, "G": 1.0, "K": 1 / 1024**2}.get(raw[-1], 0.0)
            swap_used = float(raw[:-1]) * scale
            break

    return {
        "available_gib": available,
        "wired_gib": gib("Pages wired down"),
        "compressed_gib": gib("Pages occupied by compressor"),
        "swap_used_gib": swap_used,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a command under a total-system memory watchdog."
    )
    parser.add_argument(
        "--min-free", type=float, default=48.0,
        help="kill when system available memory drops below this many GiB",
    )
    parser.add_argument(
        "--max-swap", type=float, default=8.0,
        help="kill when swap in use exceeds this many GiB",
    )
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument(
        "--trace", default="/tmp/memguard-trace.csv",
        help="where to write the sampled memory trace",
    )
    parser.add_argument(
        "--reason", default="/tmp/memguard-reason.txt",
        help="where to write the explanation if the watchdog trips",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("no command given, use: memguard.py [options] -- <command>")

    pagesize = page_size()
    total_gib = int(
        subprocess.run(
            ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True
        ).stdout.strip()
    ) / GIB
    start = memory_snapshot(pagesize)

    print(
        f"[memguard] total {total_gib:.0f} GiB, available {start['available_gib']:.0f} GiB",
        file=sys.stderr,
    )
    print(
        f"[memguard] kill if available < {args.min_free:.0f} GiB "
        f"or swap > {args.max_swap:.0f} GiB",
        file=sys.stderr, flush=True,
    )

    Path(args.reason).unlink(missing_ok=True)
    process = subprocess.Popen(command, start_new_session=True)
    group = os.getpgid(process.pid)

    trace_file = open(args.trace, "w", newline="")
    writer = csv.writer(trace_file)
    writer.writerow(
        ["timestamp", "elapsed_s", "available_gib", "wired_gib",
         "compressed_gib", "swap_used_gib"]
    )

    started = time.time()
    lowest = start["available_gib"]
    tripped = None
    try:
        while process.poll() is None:
            snapshot = memory_snapshot(pagesize)
            elapsed = time.time() - started
            writer.writerow([
                datetime.now(timezone.utc).isoformat(),
                round(elapsed, 1),
                round(snapshot["available_gib"], 2),
                round(snapshot["wired_gib"], 2),
                round(snapshot["compressed_gib"], 2),
                round(snapshot["swap_used_gib"], 2),
            ])
            trace_file.flush()
            lowest = min(lowest, snapshot["available_gib"])

            if snapshot["available_gib"] < args.min_free:
                tripped = (
                    f"available memory {snapshot['available_gib']:.1f} GiB "
                    f"fell below --min-free {args.min_free:.1f} GiB"
                )
            elif snapshot["swap_used_gib"] > args.max_swap:
                tripped = (
                    f"swap in use {snapshot['swap_used_gib']:.1f} GiB "
                    f"exceeded --max-swap {args.max_swap:.1f} GiB"
                )

            if tripped:
                message = (
                    f"[memguard] KILLING: {tripped}\n"
                    f"[memguard] after {elapsed:.0f}s, command: {' '.join(command)}\n"
                    f"[memguard] trace: {args.trace}\n"
                )
                print(message, file=sys.stderr, flush=True)
                Path(args.reason).write_text(message)
                os.killpg(group, signal.SIGTERM)
                for _ in range(50):
                    if process.poll() is not None:
                        break
                    time.sleep(0.1)
                if process.poll() is None:
                    os.killpg(group, signal.SIGKILL)
                process.wait()
                return 137

            time.sleep(args.interval)
    except KeyboardInterrupt:
        os.killpg(group, signal.SIGTERM)
        process.wait()
        return 130
    finally:
        trace_file.close()

    print(
        f"[memguard] finished in {time.time() - started:.0f}s, "
        f"lowest available {lowest:.0f} GiB",
        file=sys.stderr,
    )
    return process.returncode


if __name__ == "__main__":
    sys.exit(main())
