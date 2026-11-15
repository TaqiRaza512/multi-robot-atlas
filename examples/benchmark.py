"""Compare navigation policies across scenarios and print a results table.

    uv run examples/benchmark.py            # standard set, about a minute
    uv run examples/benchmark.py --quick    # skip the largest fleets

Every number is deterministic on a given machine, so the table can be
regenerated exactly.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable, Sequence

import numpy as np

from atlas import (
    DMPCPolicy,
    Policy,
    Robot,
    SafetyFilter,
    Simulator,
    StraightLinePolicy,
    antipodal_sphere,
    head_on,
    parallel_lanes,
)

PolicyFactory = Callable[[Sequence[Robot]], Policy]

POLICIES: dict[str, PolicyFactory] = {
    "straight": lambda robots: StraightLinePolicy(),
    "straight + filter": lambda robots: SafetyFilter(StraightLinePolicy(), robots),
    "DMPC": lambda robots: DMPCPolicy(robots),
    "DMPC + filter": lambda robots: SafetyFilter(DMPCPolicy(robots), robots),
}


def scenarios(quick: bool) -> dict[str, list[Robot]]:
    selected = {
        "parallel lanes (5)": parallel_lanes(5),
        "head-on pair": head_on(),
        "sphere swap (5)": antipodal_sphere(5),
        "sphere swap (8)": antipodal_sphere(8),
    }
    if not quick:
        selected["sphere swap (12)"] = antipodal_sphere(12)
    return selected


def path_ratio(positions: np.ndarray, robots: Sequence[Robot]) -> float:
    """Mean flown path length divided by the straight-line distance."""
    flown = np.linalg.norm(np.diff(positions, axis=0), axis=2).sum(axis=0)
    direct = np.array([np.linalg.norm(r.goal - r.start) for r in robots])
    return float(np.mean(flown / direct))


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="skip the largest fleets")
    args = parser.parse_args()

    header = (
        f"| {'Scenario':<19} | {'Policy':<17} | Reached | Contacts | Clearance | Time    "
        "| Path  | Filter use | Wall   |"
    )
    print(header)
    print("| " + " | ".join("-" * (len(cell) - 2) for cell in header.split("|")[1:-1]) + " |")
    for scenario_name, robots in scenarios(args.quick).items():
        for policy_name, factory in POLICIES.items():
            policy = factory(robots)
            start = time.perf_counter()
            result = Simulator(robots, policy).run()
            wall = time.perf_counter() - start

            reached = f"{int(result.reached_goal.sum())} / {len(robots)}"
            duration = f"{result.duration:.2f} s" if result.all_reached_goal else "timeout"
            clearance = f"{result.min_clearance:+.2f} m"
            ratio = (
                f"{path_ratio(result.positions, robots):.2f}" if result.all_reached_goal else "-"
            )
            use = (
                f"{100 * policy.intervention_rate:.1f} %"
                if isinstance(policy, SafetyFilter)
                else "-"
            )
            print(
                f"| {scenario_name:<19} | {policy_name:<17} | {reached:<7} "
                f"| {len(result.robot_collisions):<8} | {clearance:<9} | {duration:<7} "
                f"| {ratio:<5} | {use:<10} | {wall:5.1f} s |"
            )


if __name__ == "__main__":
    main()
