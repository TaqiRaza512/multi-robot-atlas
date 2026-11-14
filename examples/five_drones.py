"""Five drones swap sides of a sphere using naive straight-line flight.

The policy ignores other drones, so they collide near the centre. The
simulator detects and reports those collisions; it does not avoid them.

    uv run examples/five_drones.py                      # print a summary
    uv run examples/five_drones.py --video swap.mp4     # also render a video
    uv run mjpython examples/five_drones.py --view      # live 3D viewer (macOS needs mjpython)
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from atlas import SimulationResult, Simulator, StraightLinePolicy, antipodal_sphere


def report(sim: Simulator, result: SimulationResult) -> None:
    print(f"simulated {result.duration:.2f} s")
    for i, robot in enumerate(sim.robots):
        error = float(np.linalg.norm(result.positions[-1, i] - robot.goal))
        status = "reached goal" if result.reached_goal[i] else "DID NOT reach goal"
        print(f"  {robot.name}: {status} (final error {error:.3f} m)")
    print(f"minimum clearance between drones: {result.min_clearance:+.3f} m")
    if result.collision_free:
        print("collisions: none")
        return
    pairs = result.colliding_pairs
    print(f"drone-drone collisions: {len(result.robot_collisions)} contact(s), pairs {pairs}")
    for event in result.robot_collisions:
        print(
            f"  t = {event.time:6.3f} s  drone_{event.robot_a} x drone_{event.robot_b}  "
            f"lasted {event.duration:6.3f} s  penetration {1000 * event.max_penetration:.1f} mm"
        )
    if result.ground_collisions:
        hit = sorted({event.robot_a for event in result.ground_collisions})
        print(f"ground contacts: drones {hit}")


def run_with_viewer(sim: Simulator) -> SimulationResult:
    import mujoco.viewer

    try:
        viewer = mujoco.viewer.launch_passive(sim.model, sim.data)
    except RuntimeError as error:
        if sys.platform == "darwin":
            raise SystemExit(
                "On macOS the live viewer must run under mjpython:\n"
                "    uv run mjpython examples/five_drones.py --view"
            ) from error
        raise

    with viewer:
        sim.reset()
        while viewer.is_running() and not sim.done:
            tick = time.perf_counter()
            sim.step()
            viewer.sync()
            time.sleep(max(0.0, sim.config.policy_period - (time.perf_counter() - tick)))
    return sim.result()


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--video", metavar="PATH", help="render the run to a video file")
    parser.add_argument("--view", action="store_true", help="watch live in the MuJoCo viewer")
    args = parser.parse_args()

    sim = Simulator(antipodal_sphere(num_robots=5), StraightLinePolicy())
    result = run_with_viewer(sim) if args.view else sim.run()
    report(sim, result)

    if args.video:
        from atlas.video import save_video

        print(f"saved video to {save_video(sim, result, args.video)}")


if __name__ == "__main__":
    main()
