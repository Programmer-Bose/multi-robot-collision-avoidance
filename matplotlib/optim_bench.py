import numpy as np
from run_episode import run_episode

OPTIMIZERS = ["de", "cem", "random_shoot"]
SEEDS = [42]

def patch_planner_optimizer(optimizer_name):
    import de_mpc
    orig_init = de_mpc.DEMPCPlanner.__init__
    def new_init(self, *a, **kw):
        kw["optimizer"] = optimizer_name
        orig_init(self, *a, **kw)
    de_mpc.DEMPCPlanner.__init__ = new_init

results = {}
for opt in OPTIMIZERS:
    patch_planner_optimizer(opt)
    rows = []
    for seed in SEEDS:
        print(f"Running {opt} with seed {seed}")
        env, solve_times, _ = run_episode(seed=seed, max_steps=600, horizon=10, verbose=False, optimizer=opt)
        collided, kind = env.check_collision()
        rows.append({
            "seed": seed,
            "success": env.all_tasks_done() and not collided,
            "collided": collided,
            "steps": len(env.history["robot"]),
            "avg_solve_time": float(np.mean(solve_times)),
        })
    results[opt] = rows
    succ = sum(r["success"] for r in rows)
    avg_t = np.mean([r["avg_solve_time"] for r in rows])
    print(f"{opt}: success {succ}/{len(SEEDS)}, avg solve time {avg_t:.3f}s")