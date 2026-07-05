"""
Imitation pretraining (behavior cloning) on DE-MPC demonstration data.

This is the "teacher-student" bridge in the design: DE-MPC runs offline with
privileged ground-truth obstacle state to generate (state, action) pairs;
here we train the deployed network -- which only ever sees onboard
rangefinder + occupancy grid, exactly as recorded -- to imitate those
actions via supervised regression (max-likelihood under the actor's own
Gaussian, which is the natural BC loss for a stochastic-policy network:
it's already differentiable and reuses the exact PolicyValueNet forward
pass PPO will fine-tune later, so there's no architecture mismatch handoff).

Expected input: one or more .npz files produced by run_episode.save_dataset,
each containing arrays: robot_state, goal_dist, goal_bearing, ranges,
occupancy_grid, action.

Usage:
    python3 imitation_pretrain.py --data data_traj/*.npz --epochs 50
"""

import argparse
import glob
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

from networks import PolicyValueNet


class DemoDataset(Dataset):
    def __init__(self, npz_paths):
        ranges, occ, goal_dist, goal_bearing, actions, velocity = [], [], [], [], [], []
        for path in npz_paths:
            d = np.load(path)
            n = len(d["action"])
            ranges.append(d["ranges"])
            occ.append(d["occupancy_grid"])
            goal_dist.append(d["goal_dist"].reshape(-1, 1))
            goal_bearing.append(d["goal_bearing"].reshape(-1, 1))
            actions.append(d["action"])
            # previous action as the "velocity" feature, matching how
            # RobotNavGymEnv builds this field online (shifted by one step,
            # first step gets zeros -- keeps train/deploy features consistent)
            prev_action = np.vstack([np.zeros((1, 2)), d["action"][:-1]])
            velocity.append(prev_action)

        self.ranges = np.concatenate(ranges, axis=0).astype(np.float32)
        self.occ = np.concatenate(occ, axis=0).astype(np.float32)
        self.goal_dist = np.concatenate(goal_dist, axis=0).astype(np.float32)
        self.goal_bearing = np.concatenate(goal_bearing, axis=0).astype(np.float32)
        self.actions = np.concatenate(actions, axis=0).astype(np.float32)
        self.velocity = np.concatenate(velocity, axis=0).astype(np.float32)

        # normalize to comparable scales (occupancy_grid is already 0/1, left alone;
        # goal_bearing is already in [-pi, pi], left alone)
        self.ranges = self.ranges / 5.0          # sensor_range in rangefinder.py
        self.goal_dist = self.goal_dist / 14.0   # ~world diagonal, sqrt(10^2+10^2)
        self.velocity = self.velocity / np.array([1.0, np.pi / 2], dtype=np.float32)

        print(f"Loaded {len(self.actions)} (state, action) pairs from {len(npz_paths)} file(s)")

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        return {
            "ranges": self.ranges[idx],
            "occupancy_grid": self.occ[idx],
            "goal_dist": self.goal_dist[idx],
            "goal_bearing": self.goal_bearing[idx],
            "velocity": self.velocity[idx],
        }, self.actions[idx]


def collate(batch):
    obs_list, action_list = zip(*batch)
    obs = {k: torch.as_tensor(np.stack([o[k] for o in obs_list]), dtype=torch.float32)
           for k in obs_list[0].keys()}
    actions = torch.as_tensor(np.stack(action_list), dtype=torch.float32)
    return obs, actions


def train_bc(net, train_loader, val_loader, epochs=50, lr=1e-3, device="cpu",
             loss_type="nll"):
    """
    loss_type:
        "nll" - negative log-likelihood under the actor's Gaussian (matches
                what PPO will optimize; also trains log_std sensibly)
        "mse" - plain MSE on the mean action (simpler, ignores log_std)
    """
    net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        net.train()
        train_loss = 0.0
        for obs, actions in train_loader:
            obs = {k: v.to(device) for k, v in obs.items()}
            actions = actions.to(device)

            dist, _ = net(obs)
            if loss_type == "nll":
                loss = -dist.log_prob(actions).sum(-1).mean()
            else:
                loss = ((dist.mean - actions) ** 2).sum(-1).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * actions.shape[0]
        train_loss /= len(train_loader.dataset)

        net.eval()
        val_loss, val_mae = 0.0, 0.0
        with torch.no_grad():
            for obs, actions in val_loader:
                obs = {k: v.to(device) for k, v in obs.items()}
                actions = actions.to(device)
                dist, _ = net(obs)
                if loss_type == "nll":
                    loss = -dist.log_prob(actions).sum(-1).mean()
                else:
                    loss = ((dist.mean - actions) ** 2).sum(-1).mean()
                val_loss += loss.item() * actions.shape[0]
                val_mae += (dist.mean - actions).abs().sum().item()
        val_loss /= len(val_loader.dataset)
        val_mae /= len(val_loader.dataset)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in net.state_dict().items()}

        if epoch == 1 or epoch % max(1, epochs // 10) == 0 or epoch == epochs:
            print(f"epoch {epoch:3d}/{epochs} | train_loss={train_loss:.4f} "
                  f"val_loss={val_loss:.4f} val_action_mae={val_mae:.4f}")
    if best_state is not None:
        net.load_state_dict(best_state)
        print(f"Restored best checkpoint (val_loss={best_val_loss:.4f})")
    return net
    return net


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", nargs="+", required=True,
                         help="glob pattern(s) for .npz demo files")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--loss_type", choices=["nll", "mse"], default="nll")
    parser.add_argument("--out", default="bc_pretrained.pt")
    args = parser.parse_args()

    paths = []
    for pattern in args.data:
        paths.extend(sorted(glob.glob(pattern)))
    if not paths:
        raise FileNotFoundError(f"No .npz files matched {args.data}")

    dataset = DemoDataset(paths)
    n_val = max(1, int(0.1 * len(dataset)))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    n_rays = dataset.ranges.shape[1]
    grid_size = dataset.occ.shape[1]
    net = PolicyValueNet(n_rays=n_rays, grid_size=grid_size,
                          action_low=[-0.3, -np.pi / 2], action_high=[1.0, np.pi / 2])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_bc(net, train_loader, val_loader, epochs=args.epochs, lr=args.lr,
             device=device, loss_type=args.loss_type)

    torch.save(net.state_dict(), args.out)
    print(f"Saved pretrained weights to {args.out}")


if __name__ == "__main__":
    main()
