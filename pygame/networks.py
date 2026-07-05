"""
Custom decentralized policy/value network.

Architecture (see docstring at bottom for the 50-word version):

  1. LocalEncoder: fuses this robot's own sensing (rangefinder + occupancy
     grid + goal_dist/bearing + velocity) into a single embedding vector.
     Occupancy grid goes through a small CNN (it's spatial); everything
     else goes through an MLP; the two are concatenated and fused.

  2. NeighborAttention: multi-head attention where the query is the robot's
     own embedding and the keys/values are embeddings of nearby robots
     (variable count, padding-masked). This is what gives size-invariance
     across team sizes -- it works identically whether there are 0, 3, or
     20 neighbors. With zero neighbors (today's single-robot env) it is a
     no-op by construction (residual connection + masked-out attention).

  3. PolicyValueHead: actor (Gaussian mean + learned log_std over [v, omega])
     and critic (scalar value), both small MLPs on top of the fused
     local+social embedding.

Everything here is written for a SINGLE robot's forward pass. In the
multi-robot setting, every robot runs the same weights (fully decentralized,
no parameter sharing tricks needed since it's the same network instance
copied/broadcast to each agent) -- this file doesn't assume how many robots
there are; that only shows up in `neighbor_embeddings.shape[1]` at call time.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalEncoder(nn.Module):
    def __init__(self, n_rays=15, grid_size=21, embed_dim=128):
        super().__init__()
        self.grid_size = grid_size

        # small CNN over the occupancy grid (spatial structure matters here)
        self.grid_cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # -> 11x11
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # -> 6x6
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, grid_size, grid_size)
            grid_feat_dim = self.grid_cnn(dummy).shape[1]

        # vector branch: rangefinder + goal_dist + goal_bearing + velocity(2)
        vec_dim = n_rays + 1 + 1 + 2
        self.vec_mlp = nn.Sequential(
            nn.Linear(vec_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        self.fusion = nn.Sequential(
            nn.Linear(grid_feat_dim + 64, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, ranges, occupancy_grid, goal_dist, goal_bearing, velocity):
        """
        ranges: (B, n_rays)
        occupancy_grid: (B, grid_size, grid_size)
        goal_dist: (B, 1)
        goal_bearing: (B, 1)
        velocity: (B, 2)
        Returns: (B, embed_dim)
        """
        grid_in = occupancy_grid.unsqueeze(1)  # (B, 1, G, G)
        grid_feat = self.grid_cnn(grid_in)

        vec_in = torch.cat([ranges, goal_dist, goal_bearing, velocity], dim=-1)
        vec_feat = self.vec_mlp(vec_in)

        fused = torch.cat([grid_feat, vec_feat], dim=-1)
        return self.fusion(fused)


class NeighborAttention(nn.Module):
    """
    Single-head-generalized (via nn.MultiheadAttention) attention over a
    variable, padding-masked set of neighbor embeddings. Query = self
    embedding. Residual + layernorm so that with zero real neighbors the
    block passes the self-embedding through essentially unchanged.
    """

    def __init__(self, embed_dim=128, n_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, self_embed, neighbor_embeds, neighbor_mask=None):
        """
        self_embed: (B, embed_dim)
        neighbor_embeds: (B, N, embed_dim)  -- N can be 0
        neighbor_mask: (B, N) bool, True = PAD (ignore). None if no padding.
        Returns: (B, embed_dim)  -- self-embedding enriched with social context
        """
        if neighbor_embeds.shape[1] == 0:
            # no neighbors at all (e.g. single-robot env) -> pure pass-through
            return self.norm(self_embed)

        query = self_embed.unsqueeze(1)  # (B, 1, D)
        attn_out, _ = self.attn(query, neighbor_embeds, neighbor_embeds,
                                 key_padding_mask=neighbor_mask)
        attn_out = attn_out.squeeze(1)  # (B, D)
        return self.norm(self_embed + attn_out)  # residual


class PolicyValueNet(nn.Module):
    def __init__(self, n_rays=15, grid_size=21, embed_dim=128, n_heads=4,
                 action_dim=2, action_low=None, action_high=None):
        super().__init__()
        self.encoder = LocalEncoder(n_rays, grid_size, embed_dim)
        self.social = NeighborAttention(embed_dim, n_heads)

        self.actor_mean = nn.Sequential(
            nn.Linear(embed_dim, 64), nn.ReLU(), nn.Linear(64, action_dim)
        )
        # state-independent learned log_std (standard PPO continuous-action trick)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim) - 0.5)

        self.critic = nn.Sequential(
            nn.Linear(embed_dim, 64), nn.ReLU(), nn.Linear(64, 1)
        )

        # action bounds used to squash the raw Gaussian mean via tanh scaling
        self.register_buffer("action_low", torch.as_tensor(
            action_low if action_low is not None else [-0.3, -1.57], dtype=torch.float32))
        self.register_buffer("action_high", torch.as_tensor(
            action_high if action_high is not None else [1.0, 1.57], dtype=torch.float32))

    def _embed(self, obs, neighbor_embeds=None, neighbor_mask=None):
        self_embed = self.encoder(obs["ranges"], obs["occupancy_grid"],
                                   obs["goal_dist"], obs["goal_bearing"], obs["velocity"])
        if neighbor_embeds is None:
            neighbor_embeds = torch.zeros(self_embed.shape[0], 0, self_embed.shape[1],
                                           device=self_embed.device)
        return self.social(self_embed, neighbor_embeds, neighbor_mask)

    def forward(self, obs, neighbor_embeds=None, neighbor_mask=None):
        """Returns (dist, value). obs is a dict of batched tensors."""
        fused = self._embed(obs, neighbor_embeds, neighbor_mask)

        raw_mean = self.actor_mean(fused)
        # squash to [-1, 1] then rescale to actual action bounds
        squashed = torch.tanh(raw_mean)
        mean = self.action_low + (squashed + 1.0) * 0.5 * (self.action_high - self.action_low)

        std = torch.exp(self.actor_log_std).expand_as(mean)
        dist = torch.distributions.Normal(mean, std)

        value = self.critic(fused).squeeze(-1)
        return dist, value

    def act(self, obs, neighbor_embeds=None, neighbor_mask=None, deterministic=False):
        dist, value = self.forward(obs, neighbor_embeds, neighbor_mask)
        action = dist.mean if deterministic else dist.sample()
        action = torch.clamp(action, self.action_low, self.action_high)
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob, value

    def evaluate(self, obs, actions, neighbor_embeds=None, neighbor_mask=None):
        """Used by PPO update: recompute log_prob/entropy/value for stored actions."""
        dist, value = self.forward(obs, neighbor_embeds, neighbor_mask)
        log_prob = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy, value


if __name__ == "__main__":
    # shape smoke test, single-robot (0 neighbors) and multi-robot (3 neighbors) cases
    B, n_rays, grid_size = 4, 15, 21
    obs = {
        "ranges": torch.rand(B, n_rays) * 5.0,
        "occupancy_grid": torch.randint(0, 2, (B, grid_size, grid_size)).float(),
        "goal_dist": torch.rand(B, 1) * 10.0,
        "goal_bearing": torch.rand(B, 1) * 2 - 1,
        "velocity": torch.rand(B, 2),
    }

    net = PolicyValueNet(n_rays=n_rays, grid_size=grid_size)

    # single-robot: no neighbors
    dist, value = net(obs)
    print("single-robot (0 neighbors): action mean shape", dist.mean.shape, "value shape", value.shape)

    # multi-robot: 3 neighbors, last one padded/masked out for batch item 0
    neighbor_embeds = torch.rand(B, 3, 128)
    neighbor_mask = torch.zeros(B, 3, dtype=torch.bool)
    neighbor_mask[0, 2] = True  # robot 0 only really has 2 neighbors
    dist2, value2 = net(obs, neighbor_embeds, neighbor_mask)
    print("multi-robot (3 neighbors, 1 masked): action mean shape", dist2.mean.shape)

    action, log_prob, value3 = net.act(obs)
    print("act(): action", action.shape, "log_prob", log_prob.shape, "value", value3.shape)
