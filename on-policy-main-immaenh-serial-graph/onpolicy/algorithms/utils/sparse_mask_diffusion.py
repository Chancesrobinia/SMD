import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskDiffusionSchedule:
    def __init__(self, num_steps=50, beta_start=1e-4, beta_end=2e-2, device="cpu"):
        self.num_steps = num_steps
        self.betas = torch.linspace(beta_start, beta_end, num_steps, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, x0, t, noise):
        alpha_bar_t = self.alpha_bars.to(x0.device)[t].to(dtype=x0.dtype)
        return torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1.0 - alpha_bar_t) * noise


class SparseMaskDiffusionTeacher(nn.Module):
    """
    Gaussian diffusion teacher over continuous Top-K communication masks.

    This module is used only in PPO update auxiliary loss. It never performs
    multi-step online sampling.
    """

    def __init__(
        self,
        edge_context_dim,
        hidden_dim=128,
        num_diffusion_steps=50,
        beta_start=1e-4,
        beta_end=2e-2,
        time_embed_dim=32,
        use_orthogonal=True,
        use_ReLU=True,
        debug_shapes=False,
    ):
        super(SparseMaskDiffusionTeacher, self).__init__()
        self.num_diffusion_steps = num_diffusion_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.debug_shapes = debug_shapes
        self._debug_printed = False
        act_fn = nn.ReLU() if use_ReLU else nn.Tanh()
        self.time_emb = nn.Embedding(max(512, num_diffusion_steps), time_embed_dim)
        self.denoiser = nn.Sequential(
            nn.Linear(edge_context_dim + time_embed_dim + 1, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, 1),
        )
        self._init_weights(use_orthogonal, use_ReLU)

    def _init_weights(self, use_orthogonal, use_ReLU):
        gain = nn.init.calculate_gain('relu' if use_ReLU else 'tanh')
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if use_orthogonal:
                    nn.init.orthogonal_(m.weight, gain=gain)
                else:
                    nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, edge_context, pseudo_mask, candidate_mask):
        B, K = pseudo_mask.shape[:2]
        if K == 0:
            zero = edge_context.new_tensor(0.0)
            return {
                "mask_diffusion_loss": zero,
                "teacher_logits": edge_context.new_zeros(B, 0),
                "teacher_probs": edge_context.new_zeros(B, 0),
                "teacher_mask": edge_context.new_zeros(B, 0),
                "timesteps": torch.zeros(B, dtype=torch.long, device=edge_context.device),
                "x0": edge_context.new_zeros(B, 0),
                "x0_pred": edge_context.new_zeros(B, 0),
                "smd_debug": {},
            }

        valid = candidate_mask.to(dtype=torch.bool)
        target = 2.0 * pseudo_mask.to(dtype=edge_context.dtype) - 1.0          # [B, K]
        noise = torch.randn_like(target)
        # One diffusion timestep is sampled for each training sample, then
        # broadcast across that sample's candidate edges.
        t = torch.randint(0, self.num_diffusion_steps, (B,), device=target.device)
        schedule = MaskDiffusionSchedule(self.num_diffusion_steps, self.beta_start, self.beta_end, target.device)
        t_edges = t.unsqueeze(-1).expand(-1, K)
        x_t = schedule.add_noise(target, t_edges, noise)                       # [B, K]
        t_emb = self.time_emb(t.long().clamp(max=self.time_emb.num_embeddings - 1))
        t_emb = t_emb.unsqueeze(1).expand(-1, K, -1)
        denoise_in = torch.cat([x_t.unsqueeze(-1), edge_context, t_emb], dim=-1)
        pred_noise = self.denoiser(denoise_in).squeeze(-1)                    # [B, K]

        alpha_bar = schedule.alpha_bars.to(target.device)[t].to(dtype=target.dtype)
        sqrt_alpha_bar = torch.sqrt(alpha_bar).unsqueeze(-1)
        sqrt_one_minus = torch.sqrt(1.0 - alpha_bar).unsqueeze(-1)
        x0_pred = (x_t - sqrt_one_minus * pred_noise) / sqrt_alpha_bar.clamp_min(1e-6)

        loss = F.mse_loss(pred_noise, noise, reduction="none") * valid.to(dtype=target.dtype)
        mask_diffusion_loss = loss.sum() / valid.to(dtype=target.dtype).sum().clamp_min(1.0)

        teacher_logits = x0_pred.detach()
        teacher_probs = ((teacher_logits + 1.0) * 0.5).clamp(0.0, 1.0) * valid.to(dtype=target.dtype)
        teacher_mask = (teacher_probs > 0.5).to(dtype=target.dtype) * valid.to(dtype=target.dtype)

        if self.debug_shapes and not self._debug_printed:
            print("[SMD] pseudo_mask shape", tuple(pseudo_mask.shape))
            print("[SMD] mask_diffusion_loss", float(mask_diffusion_loss.detach().cpu()))
            self._debug_printed = True

        return {
            "mask_diffusion_loss": mask_diffusion_loss,
            "teacher_logits": teacher_logits,
            "teacher_probs": teacher_probs,
            "teacher_mask": teacher_mask,
            "timesteps": t,
            "x0": target,
            "x0_pred": x0_pred,
            "smd_debug": {},
        }
