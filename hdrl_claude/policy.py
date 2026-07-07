# policy.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MapEncoder(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )
        self.proj = nn.Linear(64 * 4 * 4, out_dim)

    def forward(self, grid):
        b = grid.shape[0]
        feat = self.cnn(grid).view(b, -1)
        return self.proj(feat)


class ObstacleAwarePolicy(nn.Module):
    def __init__(self, map_feat_dim=128, task_embed_dim=64, hidden_dim=128):
        super().__init__()
        self.map_encoder = MapEncoder(map_feat_dim)
        self.task_embed = nn.Linear(2, task_embed_dim)
        self.pos_embed = nn.Linear(2, task_embed_dim)
        self.w_embed = nn.Linear(3, task_embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=task_embed_dim, nhead=4, dim_feedforward=hidden_dim,
            batch_first=True)
        self.task_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        fused_dim = map_feat_dim + task_embed_dim * 2
        self.context_proj = nn.Linear(fused_dim, hidden_dim)

        self.pointer_query = nn.Linear(hidden_dim, task_embed_dim)

        self.cp_head = nn.Sequential(
            nn.Linear(hidden_dim + task_embed_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 6),
        )
        self.cp_log_std = nn.Parameter(torch.full((3, 2), -0.5))

    def forward(self, grid, current_pos, task_points, available_mask, w, task_idx=None):
        map_feat = self.map_encoder(grid)
        pos_feat = self.pos_embed(current_pos)
        w_feat = self.w_embed(w)
        task_feat = self.task_embed(task_points)
        task_feat = self.task_encoder(task_feat)

        fused = torch.cat([map_feat, pos_feat, w_feat], dim=-1)
        context = F.relu(self.context_proj(fused))

        query = self.pointer_query(context).unsqueeze(1)
        task_logits = torch.bmm(query, task_feat.transpose(1, 2)).squeeze(1)
        task_logits = task_logits.masked_fill(~available_mask, float("-inf"))

        if task_idx is None:  # inference / greedy
            chosen_idx = torch.argmax(F.softmax(task_logits, dim=-1), dim=-1)
        else:                 # training: use the task actually sampled
            chosen_idx = task_idx

        chosen_task_feat = task_feat[torch.arange(task_feat.size(0)), chosen_idx]
        goal_pos = task_points[torch.arange(task_points.size(0)), chosen_idx]  # (B,2)

        cp_input = torch.cat([context, chosen_task_feat], dim=-1)
        cp_offset = self.cp_head(cp_input).view(-1, 3, 2)  # small delta

        # interpolate straight line start->goal at t=0.25,0.5,0.75, add learned offset
        t = torch.tensor([0.25, 0.5, 0.75], device=current_pos.device).view(1, 3, 1)
        line_pts = current_pos.unsqueeze(1) * (1 - t) + goal_pos.unsqueeze(1) * t
        cp_mean = line_pts + cp_offset

        cp_std = torch.exp(self.cp_log_std).unsqueeze(0).expand_as(cp_mean)
        return task_logits, cp_mean, cp_std, chosen_idx
    
if __name__ == "__main__":

    policy = ObstacleAwarePolicy()
    grid = torch.rand(1, 1, 64, 64)
    current_pos = torch.rand(1, 2)
    task_points = torch.rand(1, 3, 2)
    available_mask = torch.tensor([[True, False, True]])
    w = torch.tensor([[0.6, 0.2, 0.2]])

    logits, cps, idx = policy(grid, current_pos, task_points, available_mask, w)
    print(logits.shape, cps.shape, idx)  # torch.Size([1,3]) torch.Size([1,3,2]) tensor([...])