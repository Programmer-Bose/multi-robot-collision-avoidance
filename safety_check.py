"""
Multi-seed safety check: runs the DE-MPC planner on several randomized
scenarios and reports whether it completes all tasks collision-free.

Usage:
    python3 safety_check.py --n_seeds 5
"""
import argparse
import numpy as np

from run_episode import run_episode, plot_trajectory


def main(n_seeds=5, max_steps=600, horizon=10):
    results = []
    for seed in range(n_seeds):
        print(f"\n=== Seed {seed} ===")
        env, solve_times = run_episode(seed=seed, max_steps=max_steps,
                                        horizon=horizon, verbose=True)
        collided, kind = env.check_collision()
        success = env.all_tasks_done() and not collided
        results.append({
            "seed": seed,
            "success": success,
            "collided": collided,
            "collision_kind": kind,
            "steps": len(env.history["robot"]),
            "avg_solve_time": float(np.mean(solve_times)),
        })
        plot_trajectory(env, save_path=f"episode_seed{seed}.png")

    print("\n===== SUMMARY =====")
    n_success = sum(r["success"] for r in results)
    for r in results:
        status = "OK" if r["success"] else f"FAIL ({r['collision_kind']})"
        print(f"seed {r['seed']}: {status} | steps={r['steps']} | "
              f"avg solve={r['avg_solve_time']:.3f}s")
    print(f"\nSuccess rate: {n_success}/{n_seeds}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=600)
    parser.add_argument("--horizon", type=int, default=10)
    args = parser.parse_args()
    main(n_seeds=args.n_seeds, max_steps=args.max_steps, horizon=args.horizon)
