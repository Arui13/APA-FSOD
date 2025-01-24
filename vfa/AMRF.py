import torch
import torch.nn as nn
import torch.nn.functional as F
from mmfewshot.detection.models import AGGREGATORS


@AGGREGATORS.register_module()
class AMRF(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AMRF, self).__init__()
        self.W_1 = nn.Conv2d(input_dim * 5, hidden_dim, kernel_size=1)
        self.MLP = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        self.W_a = nn.Linear(input_dim, input_dim, bias=False)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.ln = nn.LayerNorm(hidden_dim)

    def forward(self, f_roi, z):
        prod = f_roi * z
        diff = f_roi - z
        attn_weights = F.softmax(torch.matmul(f_roi.squeeze(-1).squeeze(-1),
                                              self.W_a(z.squeeze(-1).squeeze(-1)).T), dim=-1)
        attn = attn_weights.unsqueeze(-1).unsqueeze(-1) * f_roi
        f_roi_z = torch.cat([f_roi, z.expand_as(f_roi)], dim=1)
        concat_features = torch.cat([f_roi_z, prod, diff, attn], dim=1)
        f_prime = self.W_1(concat_features)
        f_prime_flat = f_prime.view(f_prime.size(0), -1)
        f_prime_norm = self.ln(f_prime_flat)
        f_transformed = self.MLP(f_prime_norm)
        f_out = self.alpha * f_transformed.unsqueeze(-1).unsqueeze(-1) + (1 - self.alpha) * f_roi
        return f_out
